import torch
from torch.utils.data import DataLoader
import numpy as np
from baseline_model import BaselineVAE, vae_loss
import pandas as pd
import selfies as sf
from sklearn.model_selection import train_test_split

# Data prep
df = pd.read_csv('../smiles_selfies_full.csv')
df['tokens'] = df['selfies'].apply(lambda x: list(sf.split_selfies(x)))

all_tokens =  [tok for seq in df['tokens'] for tok in seq]
vocab = sorted(set(all_tokens))
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
vocab = [PAD, SOS, EOS] + vocab

tok2id = {tok: idx for idx, tok in enumerate(vocab)}
id2tok = {idx: tok for tok, idx in tok2id.items()}

def tokens_to_ids(tokens, tok2id):
    return np.array([tok2id[t] for t in tokens])
df['token_ids'] = df['tokens'].apply(lambda toks: tokens_to_ids(toks, tok2id))

sequences = df['token_ids'].tolist()
max_len = max(len(seq) for seq in sequences)
padded_data = np.zeros((len(sequences), max_len), dtype=sequences[0].dtype)

for i, seq in enumerate(sequences):
    padded_data[i, :len(seq)] = seq

data = padded_data
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
val_data, test_data = train_test_split(data, test_size=0.5, random_state=42, shuffle=True)

# Grid Search
device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_sizes = [64, 128, 256, 512, 1024]
betas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

batch_size = 1024
max_epochs = 50
patience = 10

lr_factor = 0.5
min_lr = 1e-5

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

for latent_size in latent_sizes:
    embed_size = min(256, max(128, latent_size // 2))
    hidden_size = 2 * latent_size

    for beta in betas:
        print(f"\n ----Training latent_dim={latent_size}, hidden={hidden_size}, embed={embed_size}, beta={beta}----")

        model = BaselineVAE(vocab_size=len(vocab), max_len=train_data.shape[-1], embed_size=embed_size, hidden_size=hidden_size, latent_size=latent_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, min_lr=min_lr, verbose=True)

        best_val_loss_config = float('inf')
        epochs_no_improve = 0

        for epoch in range(1, max_epochs+1):
            model.train()
            total_loss = 0
            val_loss = 0
            for x in train_loader:
                x = x.to(device)
                logits, mu, logvar = model(x)
                loss, rec, kl = vae_loss(logits, x, mu, logvar, beta=beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss
        
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device)
                    logits, mu, logvar = model(x)
                    loss, rec, kl = vae_loss(logits, x, mu, logvar, beta)
                    val_loss += loss.item()

            total_loss = total_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch:03d} | total_loss={total_loss:.4f} | val_loss={val_loss:.4f}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss_config:
                best_val_loss_config = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
            if epochs_no_improve > patience:
                print(f"Early stopping after {epoch} epochs")
                break

        torch.save({
            'latent_size': latent_size,
            'hidden_size': hidden_size,
            'embed_size': embed_size,
            'beta': beta,
            'state_dict': best_model_state,
            'val_loss': best_val_loss_config
        }, f"trained_models/vae_lat{latent_size}_beta{beta}.pt")