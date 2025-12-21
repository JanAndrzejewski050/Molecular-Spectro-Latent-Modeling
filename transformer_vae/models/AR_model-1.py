import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_size):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.dim() == 2:
            B, T = x.shape
        else:
            B, T, _ = x.shape
        pos = self.pe[:T]                 
        pos = pos.unsqueeze(0)            
        pos = pos.expand(B, T, -1)        # [B, T, H]
        return pos


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

    def forward(self, x, mask=None, causal=False):   # [B, T, H]
        B, T, H = x.shape
        Q = self.W_q(x)     # [B, T, num_heads * H]
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(B, self.num_heads, T, H) # [B, A, T, H]
        K = K.view(B, self.num_heads, T, H)
        V = V.view(B, self.num_heads, T, H)

        attn_logits = torch.einsum('baih,bajh->baij', Q, K)    # [B, A, T, H] @ [B, A, H, T] = [B, A, T, T]

        if causal:
            causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=x.device), diagonal=1)
            attn_logits = attn_logits.masked_fill(causal_mask[None, None], float("-inf"))

        attn = F.softmax(attn_logits / (H ** 0.5), dim=-1)   
        h = torch.einsum('baij,bajh->baih',attn, V)  # [B, A, T, H]
        h = h.view(B, T, self.num_heads*H)  # [B, T, A*H]
        h = self.W_o(h)     # [B, T, H]
        
        h = self.norm1(x + h)
        h = self.norm2(h + self.ff(h))
        return h


class VaeTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_size, max_len, attn_heads=8):
        super().__init__()
        # Encoder
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size // 2)
        self.pos_encoder = PositionalEmbedding(max_len, hidden_size // 2)
        # Self Attention on embeddings
        self.encoder = MultiHeadAttention(hidden_size, attn_heads)
        # Attention Pooling matrix
        self.W_pool = nn.Linear(hidden_size, hidden_size)
        # vae heads
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.max_len = max_len
        self.fc_z2h = nn.Linear(latent_size + hidden_size, hidden_size)
        # Self attention on masked sequences
        self.decoder = MultiHeadAttention(hidden_size, num_heads=attn_heads)
        # output head
        self.fc_output = nn.Linear(hidden_size, vocab_size)

    def encode(self, x):  
        emb = self.embedding(x)    # [B, T, H // 2]
        pos_encoding = self.pos_encoder(x)
        emb_pos = torch.cat([emb, pos_encoding], dim=-1) 
        h = self.encoder(emb_pos)
        h = torch.mean(h, dim=-2)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, emb_pos
    
    def decode_train(self, z, emb_pos):
        B, T, H = emb_pos.shape
        z_expand = z.unsqueeze(1).expand(B, T, z.size(-1))
        h = torch.cat([emb_pos, z_expand], dim=-1)  # [B, T, H+d]
        h = self.fc_z2h(h)
        h = self.decoder(h, causal=True)
        logits = self.fc_output(h)
        return logits
    
    def decode(self, z, sos_id=1):
        B, _ = z.shape
        device = z.device
        generated = []
        sos = torch.full((B, 1), sos_id, device=device, dtype=torch.long)   # [B, 1]
        curr_h = self.embedding(sos)
        generated.append(sos)

        for i in range(self.max_len):
            pos = self.pos_encoder(curr_h)
            pos_emb = torch.cat([curr_h, pos], dim=-1)
            T = curr_h.size(1)
            z_h = z.unsqueeze(1).expand(B, T, -1)
            h = self.fc_z2h(torch.cat([pos_emb, z_h], dim=-1))
            x = self.decoder(h, causal=True)
            
            last = x[:, -1:, :]              # [B, 1, H]
            logits = self.fc_output(last)    # [B, 1, V]
            next_token = torch.argmax(logits, dim=-1)  # [B, 1]
            generated.append(next_token)

            curr_emb = self.embedding(next_token)
            curr_h = torch.cat([curr_h, curr_emb], dim=1)  # [B, i, H]

        gen = torch.cat(generated, dim=1)  # [B, max_len+1]
        return gen

    def forward(self, x, mode='eval'):
        mu, logvar, emb = self.encode(x)
        std = torch.exp(logvar)
        eps = torch.randn_like(mu)
        z = mu + std * eps
        if mode == 'train':
            logits = self.decode_train(z, emb)
            return logits, mu, logvar
        else:
            tokens = self.decode(z)
            return tokens, mu, logvar


def vae_loss(logits, x, mu, logvar, beta=0.01, pad_id=0):
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

