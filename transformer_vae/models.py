import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Slow version
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.d = hidden_size // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):   # [B, T, H]
        B, T, H = x.shape
        Q = self.W_q(x)     # [B, T, num_heads * d]
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(B, self.num_heads, T, self.d) # [B, A, T, d]
        K = K.view(B, self.num_heads, T, self.d)
        V = V.view(B, self.num_heads, T, self.d)
        attn_logits = torch.einsum('batd,badT->batT', Q, K.transpose(2,3))    # [B, A, T, d] @ [B, A, d, T] = [B, A, T, T]
        attn = F.softmax(attn_logits / (self.d ** 0.5), dim=-1)   
        h = torch.einsum('batt,bath->bath',attn, V)  # [B, A, T, d]
        h = h.view(B, T, self.num_heads*self.d)  # [B, T, A*d]
        h = self.W_o(h)     # [B, T, d]
        return h
    

# Fast scnd version
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, p=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.d = hidden_size // num_heads
        self.num_heads = num_heads
        self.W_qkv = nn.Linear(hidden_size, 3*hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size,hidden_size*2),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p)

    def forward(self, x):   # [B, T, H]
        B, T, H = x.shape
        qkv = self.W_qkv(x)     # [B, T, num_heads * d]
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.reshape(B, self.num_heads, T, self.d) # [B, A, T, d]
        K = K.reshape(B, self.num_heads, T, self.d)
        V = V.reshape(B, self.num_heads, T, self.d)

        attn_logits = torch.einsum('batd,badT->batT', Q, K.transpose(2,3))    # [B, A, T, d] @ [B, A, d, T] = [B, A, T, T]
        attn = F.softmax(attn_logits / (self.d ** 0.5), dim=-1)   
        h = torch.einsum('batt,bath->bath',attn, V)  # [B, A, T, d]
        h = h.view(B, T, self.num_heads*self.d)  # [B, T, A*d]

        h = self.W_o(h)    # [B, T, d]
        h = self.norm1(x + self.dropout(h))
        h = self.fc(h)
        h = self.norm2(x + self.dropout(h))
        return h
    

# Version that worked well (for some reason)
    


# Correct transformer version (faster heads)
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.d = hidden_size
        self.num_heads = num_heads
        self.W_q = nn.Linear(hidden_size, num_heads * hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, num_heads * hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, num_heads * hidden_size, bias=False)
        self.W_o = nn.Linear(num_heads * hidden_size, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.ReLU(),
            nn.Linear(4*hidden_size, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):   # [B, T, H]
        B, T, H = x.shape
        Q = self.W_q(x)     # [B, T, num_heads * H]
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(B, self.num_heads, T, H) # [B, A, T, H]
        K = K.view(B, self.num_heads, T, H)
        V = V.view(B, self.num_heads, T, H)

        attn_logits = torch.einsum('bath,bahT->batT', Q, K.transpose(2,3))    # [B, A, T, H] @ [B, A, H, T] = [B, A, T, T]
        attn = F.softmax(attn_logits / (H ** 0.5), dim=-1)   
        h = torch.einsum('batt,bath->bath',attn, V)  # [B, A, T, H]
        h = h.view(B, T, self.num_heads*H)  # [B, T, A*H]
        h = self.W_o(h)     # [B, T, H]
        
        h = self.norm1(x + h)
        h = self.norm2(h + self.ff(h))
        return h
    

# Pos encoders
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_size):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, hidden_size)

    def forward(self, x):
        if x.dim() == 2:
            B, T = x.shape
        elif x.dim() == 3:
            B, T, _ = x.shape
        positions = torch.arange(0, T, 1, device=x.device).unsqueeze(0)
        return self.pos_emb(positions)


class VaeTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_size, max_len, attn_heads=10, mask_token_id=0, seed_size=5):
        super().__init__()
        # Encoder
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = PositionalEmbedding(max_len, hidden_size)
        # Self Attention on embeddings
        self.encoder = MultiHeadAttention(hidden_size, attn_heads)
        # Attention Pooling matrix
        self.W_pool = nn.Linear(hidden_size, hidden_size)
        # vae heads
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.max_len = max_len
        self.seed_size = seed_size
        self.fc_seed = nn.Linear(latent_size, seed_size * hidden_size)
        # Self attention on masked sequences
        self.decoder = MultiHeadAttention(hidden_size, attn_heads)
        # output head
        self.fc_output = nn.Linear(hidden_size, vocab_size)

    def encode(self, x):  
        h = self.embedding(x)    # [B, T, H]
        pos_encoding = self.pos_encoder(x)
        h += pos_encoding
        h = self.encoder(h)
        h = torch.mean(h, dim=-2)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        # for now we will mask to the full sequence size (for now mask is zero vector)
        B, _ = z.shape
        L, H, S = self.max_len, self.hidden_size, self.seed_size
        x = torch.zeros((B, L, H), device=z.device)
        idxs = torch.linspace(0, L-1, S, device=z.device).long()
        seeds = self.fc_seed(z).view(B, self.seed_size, H)
        x[:, idxs, :] = seeds
        pos_encoding = self.pos_encoder(x)
        x += pos_encoding
        h = self.decoder(x)
        logits = self.fc_output(h)
        return logits
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(logvar)
        eps = torch.randn_like(mu)
        z = mu + std * eps
        logits = self.decode(z)
        return logits, mu, logvar
    
def vae_loss(logits, x, mu, logvar, beta=0.01, pad_id=1):
    B, T, V = logits.shape
    logits_flat = logits.reshape((B*T, V))
    targets_flat = x.reshape(B*T)

    mask = (targets_flat != pad_id)
    valid_logits = logits_flat[mask]
    valid_targets = targets_flat[mask]

    rec_loss = F.cross_entropy(valid_logits, valid_targets)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = rec_loss + beta * kl_loss
    return loss, rec_loss, kl_loss