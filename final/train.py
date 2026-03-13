import torch
from torch.utils.data import DataLoader
import numpy as np
from model import VaeTransformer, vae_loss
import pandas as pd
import selfies as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import wandb
import os
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

@torch.no_grad()
def accuracy(model, loader, pad_id=0, device='cuda'):
    model.eval()
    total_tok = total_correct = total_seq = perfect = 0
    for x in loader:
        x = x.to(device)
        logits, _, _, _ = model(x, mode='eval')
        for i, logit in enumerate(logits):
            pred = logit.argmax(-1)      # [L_pred]
            true = x[i]
            true_len = (true != pad_id).sum().item()
            if true_len == 0:
                continue
            if len(pred) < true_len:
                pad = torch.full(
                    (true_len - len(pred),),
                    pad_id,
                    device=pred.device,
                    dtype=pred.dtype
                )
                pred = torch.cat([pred, pad], dim=0)
            else:
                pred = pred[:true_len]
            true = true[:true_len]
            correct = (pred == true)
            total_correct += correct.sum().item()
            total_tok += true_len
            perfect += int(correct.all())
            total_seq += 1
    return (total_correct / max(total_tok, 1),perfect / max(total_seq, 1))

def main():
    parser = argparse.ArgumentParser(description='Train VAE with specific hyperparameters')
    parser.add_argument("--latent_size", type=int, required=True, help='Latent dimension size')
    parser.add_argument("--hidden_size", type=int, required=True, help="Hidden size")
    parser.add_argument("--layers", type=int, default=1, help="Number of transformer layers (both on encoder and decoder)")
    parser.add_argument("--beta", type=float, required=True, help="Beta parameter for VAE loss")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5, help="Factor to reduce LR on plateau")
    parser.add_argument("--lr_patience", type=int, default=4, help="Patience for learning rate reduction")
    parser.add_argument("--early_stopping_patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_path", type=str, default="./data/smiles_selfies_full.csv", help="Path to CSV data")
    parser.add_argument("--project_name", type=str, default="transormer-vae", help="WandB project name")
    parser.add_argument("--save_model", type=bool, default=True, help="Whether to save the trained model")
    
    args = parser.parse_args()

    set_seed(args.seed)

    # Initialize wandb
    #wandb.init(entity='molecular-latent-space', project=args.project_name, config=vars(args))

    # Data prep
    print('Loading data...')
    df = pd.read_csv(args.data_path)
    df['tokens'] = df['selfies'].apply(lambda x: list(sf.split_selfies(x)))

    all_tokens =  [tok for seq in df['tokens'] for tok in seq]
    vocab = sorted(set(all_tokens))
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    vocab = [PAD, SOS, EOS] + vocab

    tok2id = {tok: idx for idx, tok in enumerate(vocab)}
    
    def full_molecule_tokens_to_ids(tokens, tok2id):
        # Add SOS and EOS tokens
        return np.array([1] + [tok2id[t] for t in tokens] + [2])
    
    df['token_ids'] = df['tokens'].apply(lambda toks: full_molecule_tokens_to_ids(toks, tok2id))

    sequences = df['token_ids'].tolist()
    max_len = max(len(seq) for seq in sequences)
    padded_data = np.zeros((len(sequences), max_len), dtype=sequences[0].dtype)

    for i, seq in enumerate(sequences):
        padded_data[i, :len(seq)] = seq

    data = padded_data
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=args.seed, shuffle=True)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed, shuffle=True)

    print(f"Data shapes: Train {train_data.shape}, Val {val_data.shape}, Test {test_data.shape}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Move data to device
    train_data = torch.tensor(train_data, dtype=torch.long).to(device)
    val_data = torch.tensor(val_data, dtype=torch.long).to(device)
    # test_data = torch.tensor(test_data, dtype=torch.long).to(device) # Not used in training loop

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training with latent_dim={args.latent_size}, hidden={args.hidden_size}, layers={args.layers}, beta={args.beta}")

    model = VaeTransformer(
        vocab_size=len(vocab), 
        max_len=train_data.shape[-1], 
        hidden_size=args.hidden_size, 
        latent_size=args.latent_size,
        layers=args.layers
    ).to(device)

    #wandb.watch(model, log="all", log_freq=100)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=args.lr_patience, min_lr=1e-7, verbose=True,
        factor=args.lr_reduce_factor
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Create directory for models if it doesn't exist
    os.makedirs("trained_models", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_rec = 0
        total_kl = 0
        total_len = 0
        
        for x in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
            x = x.to(device)
            logits, mu, logvar, pred_len = model(x)
            loss, rec, kl, len_loss = vae_loss(logits, x, mu, logvar, pred_len, beta=args.beta)
            
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_rec += rec.item()
            total_kl += kl.item()
            total_len += len_loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        avg_train_rec = total_rec / len(train_loader)
        avg_train_kl = total_kl / len(train_loader)
        avg_train_len = total_len / len(train_loader)
        train_token_acc, train_seq_acc = accuracy(model, train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_rec = 0
        val_kl = 0
        val_len = 0
        
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                logits, mu, logvar, pred_len = model(x, mode='train')
                loss, rec, kl, len_loss = vae_loss(logits, x, mu, logvar, pred_len, args.beta)
                val_loss += loss.item()
                val_rec += rec.item()
                val_kl += kl.item()
                val_len += len_loss.item()


        avg_val_loss = val_loss / len(val_loader)
        avg_val_rec = val_rec / len(val_loader)
        avg_val_kl = val_kl / len(val_loader)
        avg_val_len = val_len / len(val_loader)
        val_token_acc, val_seq_acc = accuracy(model, val_loader)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Seq Acc: {train_seq_acc:.4f} | Val Seq Acc: {val_seq_acc:.4f}")
        
        # # Log to wandb
        # wandb.log({
        #     "loss/train_total": avg_train_loss,
        #     "loss/train_recovery": avg_train_rec,
        #     "loss/train_kl_divergence": avg_train_kl,
        #     "loss/val_total": avg_val_loss,
        #     "loss/val_recovery": avg_val_rec,
        #     "loss/val_kl_divergence": avg_val_kl,
            
        #     "accuracy/train_token": avg_train_token_acc,
        #     "accuracy/train_molecule": avg_train_seq_acc,
        #     "accuracy/val_token": avg_val_token_acc,
        #     "accuracy/val_molecule": avg_val_seq_acc,
            
        #     "epoch": epoch,
        #     "lr": optimizer.param_groups[0]['lr']
        # })

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            data_to_save = {
                'latent_size': args.latent_size,
                'hidden_size': args.hidden_size,
                'layers': args.layers,
                'beta': args.beta,
                'val_loss': best_val_loss
            }
            if args.save_model:
                data_to_save['state_dict'] = model.state_dict()
            torch.save(data_to_save, f"trained_models/vae_lat{args.latent_size}_beta{args.beta}.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve > args.early_stopping_patience:
            print(f"Early stopping after {epoch} epochs")
            break
            
    wandb.finish()

if __name__ == "__main__":
    main()