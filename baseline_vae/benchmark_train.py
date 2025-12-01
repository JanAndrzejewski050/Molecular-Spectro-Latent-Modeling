import torch
from torch.utils.data import DataLoader
import numpy as np
from baseline_model import BaselineVAE, vae_loss
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
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description="Train Baseline VAE with specific hyperparameters")
    parser.add_argument("--latent_size", type=int, required=True, help="Latent dimension size")
    parser.add_argument("--beta", type=float, required=True, help="Beta parameter for VAE loss")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hidden_size", type=int, default=None, help="Hidden size (default: 2*latent)")
    parser.add_argument("--embed_size", type=int, default=None, help="Embedding size (default: logic based on latent)")
    parser.add_argument("--data_path", type=str, default="./data/smiles_selfies_full.csv", help="Path to CSV data")
    parser.add_argument("--project_name", type=str, default="molecular-latent-space", help="WandB project name")
    
    args = parser.parse_args()

    set_seed(args.seed)

    # Initialize wandb
    wandb.init(entity='casus-mala', project=args.project_name, config=vars(args))
    
    # Data prep
    print("Loading data...")
    df = pd.read_csv(args.data_path)
    df['tokens'] = df['selfies'].apply(lambda x: list(sf.split_selfies(x)))

    all_tokens =  [tok for seq in df['tokens'] for tok in seq]
    vocab = sorted(set(all_tokens))
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    vocab = [PAD, SOS, EOS] + vocab

    tok2id = {tok: idx for idx, tok in enumerate(vocab)}
    
    def tokens_to_ids(tokens, tok2id):
        return np.array([tok2id[t] for t in tokens])
    
    df['token_ids'] = df['tokens'].apply(lambda toks: tokens_to_ids(toks, tok2id))

    sequences = df['token_ids'].tolist()
    max_len = max(len(seq) for seq in sequences)
    padded_data = np.zeros((len(sequences), max_len), dtype=sequences[0].dtype)

    for i, seq in enumerate(sequences):
        padded_data[i, :len(seq)] = seq

    data = padded_data
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=args.seed, shuffle=True)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=args.seed, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Move data to device
    # train_data = torch.tensor(train_data, dtype=torch.long).to(device)
    # val_data = torch.tensor(val_data, dtype=torch.long).to(device)
    # test_data = torch.tensor(test_data, dtype=torch.long).to(device) # Not used in training loop

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Model setup
    # Logic from benchmark_gridsearch.py
    embed_size = args.embed_size if args.embed_size else min(256, max(128, args.latent_size // 2))
    hidden_size = args.hidden_size if args.hidden_size else 2 * args.latent_size
    
    print(f"Training with latent_dim={args.latent_size}, hidden={hidden_size}, embed={embed_size}, beta={args.beta}")

    model = BaselineVAE(
        vocab_size=len(vocab), 
        max_len=train_data.shape[-1], 
        embed_size=embed_size, 
        hidden_size=hidden_size, 
        latent_size=args.latent_size
    ).to(device)

    # wandb.watch(model, log="all", log_freq=100)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, min_lr=1e-5, verbose=True)

    best_val_loss = float('inf')
    patience = 10
    epochs_no_improve = 0
    
    # Create directory for models if it doesn't exist
    os.makedirs("trained_models", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_rec = 0
        total_kl = 0
        
        for x in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
            x = x.to(device)
            logits, mu, logvar = model(x)
            loss, rec, kl = vae_loss(logits, x, mu, logvar, beta=args.beta)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_rec += rec.item()
            total_kl += kl.item()
        
        print(f"Epoch {epoch:03d} loss {loss.item():.4f}")
    
        avg_train_loss = total_loss / len(train_loader)
        avg_train_rec = total_rec / len(train_loader)
        avg_train_kl = total_kl / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_rec = 0
        val_kl = 0
        
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                logits, mu, logvar = model(x)
                loss, rec, kl = vae_loss(logits, x, mu, logvar, args.beta)
                val_loss += loss.item()
                val_rec += rec.item()
                val_kl += kl.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_rec = val_rec / len(val_loader)
        avg_val_kl = val_kl / len(val_loader)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # # Log to wandb
        # wandb.log({
        #     "epoch": epoch,
        #     "train_loss": avg_train_loss,
        #     "train_rec_loss": avg_train_rec,
        #     "train_kl_loss": avg_train_kl,
        #     "val_loss": avg_val_loss,
        #     "val_rec_loss": avg_val_rec,
        #     "val_kl_loss": avg_val_kl,
        #     "lr": optimizer.param_groups[0]['lr']
        # })

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({
                'latent_size': args.latent_size,
                'hidden_size': hidden_size,
                'embed_size': embed_size,
                'beta': args.beta,
                'state_dict': model.state_dict(),
                'val_loss': best_val_loss
            }, f"trained_models/vae_lat{args.latent_size}_beta{args.beta}.pt")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve > patience:
            print(f"Early stopping after {epoch} epochs")
            break
            
    wandb.finish()

if __name__ == "__main__":
    main()
