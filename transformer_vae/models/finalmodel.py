import torch 
import torch.nn.functional as F
import math
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.dim() == 3:
            B, T, _ = x.shape
        elif x.dim() == 2:
            B, T = x.shape
        if T <= self.pe.size(0):
            pe = self.pe[:T]  
        else:
            device = x.device
            H = self.hidden_size
            position = torch.arange(T, dtype=torch.float, device=device).unsqueeze(1)  
            div_term = torch.exp(torch.arange(0, H, 2, device=device).float() * (-math.log(10000.0)/H))
            pe = torch.zeros(T, H, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.d = hidden_size // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(4*hidden_size, hidden_size),
            nn.Dropout(p=0.1)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v, pad_mask=None, causal=False):   # [B, T, H]
        B, T_q, H = q.shape
        _, T_v, _ = v.shape
        Q = self.W_q(q)     # [B, T, num_heads * H]
        K = self.W_k(k)
        V = self.W_v(v)
        Q = Q.view(B, self.num_heads, T_q, self.d) # [B, A, T, H]
        K = K.view(B, self.num_heads, T_v, self.d)
        V = V.view(B, self.num_heads, T_v, self.d)

        attn_logits = torch.einsum('baih,bajh->baij', Q, K)    # [B, A, T, H] @ [B, A, H, T] = [B, A, T, T]

        if pad_mask is not None:
            key_mask = pad_mask[:, None, None, :]  # [B,1,1,T_k]
            attn_logits = attn_logits.masked_fill(~key_mask, float('-inf'))

        attn = F.softmax(attn_logits / math.sqrt(self.d), dim=-1)

        if pad_mask is not None:
            query_mask = pad_mask[:, None, :, None]  # [B,1,T_q,1]
            attn = attn * query_mask.float()

        h = torch.einsum('baij,bajh->baih',attn, V)  # [B, A, T, H]
        h = h.view(B, T_q, H)  # [B, T, A*H]
        h = self.W_o(h)     # [B, T, H]
        
        h = self.norm1(q + h)
        h = self.norm2(h + self.ff(h))
        return h
    
def phi(x):
    return F.elu(x) + 1

class LinearMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.num_heads = num_heads
        self.d = hidden_size // num_heads

        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, q, k, v, mask=None):
        B, Tq, H = q.shape
        _, Tk, _ = k.shape

        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        Q = Q.view(B, Tq, self.num_heads, self.d).transpose(1, 2)
        K = K.view(B, Tk, self.num_heads, self.d).transpose(1, 2)
        V = V.view(B, Tk, self.num_heads, self.d).transpose(1, 2)
        Q = phi(Q)
        K = phi(K)

        if mask is not None:
            mask = mask[:, None, :, None]  # [B, 1, T, 1]
            #Q = Q * mask
            K = K * mask
            V = V * mask

        K_sum = K.sum(dim=2)                      # [B, heads, d]
        KV = torch.einsum('bhtd,bhtv->bhdv', K, V)  # [B, heads, d, d]
        Z = 1 / (torch.einsum('bhtd,bhd->bht', Q, K_sum) + 1e-6)

        out = torch.einsum('bhtd,bhdv,bht->bhtv', Q, KV, Z)
        out = out.transpose(1, 2).contiguous().view(B, Tq, H)
        out = self.W_o(out)
        return out

class MultiSlotPooling(nn.Module):
    def __init__(self, hidden_size, num_slots):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_slots, hidden_size))

    def forward(self, h, mask):
        # h: [B, T, D]
        # mask: [B, T]
        attn = torch.einsum("kd,btd->bkt", self.queries, h)
        attn = attn.masked_fill(~mask[:, None, :], -1e9)
        attn = F.softmax(attn, dim=-1)
        slots = torch.einsum("bkt,btd->bkd", attn, h)
        return slots

class LinearSlotPooling(nn.Module):
    def __init__(self, hidden_size, num_slots):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_slots, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)

    def phi(self, x):
        return F.elu(x) + 1.0

    def forward(self, h, mask):
        Q = self.phi(self.queries)                 # [K, H]
        H = self.phi(h)                            # [B, T, H]

        w = torch.einsum("kh,bth->bkt", Q, H)
        w = w * mask[:, None, :]                   # mask padding

        slots = torch.einsum("bkt,bth->bkh", w, h)

        Z = w.sum(dim=-1, keepdim=True) + 1e-6
        slots = slots / Z

        return self.norm(slots)

class VaeTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_size, max_len, attn_heads=8, num_slots=8, layers=1, mask_token_id=0, seed_size=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoder = PositionalEmbedding(max_len, hidden_size)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([MultiHeadAttention(hidden_size, attn_heads) for _ in range(layers)])
        #self.pool = MultiSlotPooling(hidden_size, num_slots=num_slots)
        self.pool = LinearSlotPooling(hidden_size, num_slots=num_slots)

        # VAE heads
        self.fc_mu = nn.Linear(num_slots * hidden_size, latent_size)
        self.fc_logvar = nn.Linear(num_slots * hidden_size, latent_size)
        
        # Decoder
        self.max_len = max_len
        self.fc_z2h = nn.Linear(latent_size, num_slots * hidden_size)
        self.fc_lenght = nn.Sequential(
            nn.Linear(latent_size, latent_size), 
            nn.GELU(), 
            nn.Linear(latent_size, latent_size//2),
            nn.GELU(),
            nn.Linear(latent_size // 2, 1)
        )
        #self.cross_block = nn.MultiheadAttention(hidden_size, attn_heads, batch_first=True)
        self.cross_block = LinearMultiHeadAttention(hidden_size, attn_heads)
        self.decoder_blocks = nn.ModuleList([MultiHeadAttention(hidden_size, num_heads=attn_heads) for _ in range(layers)])
        
        # Output head
        self.fc_output = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, x):  
        B, _ = x.shape
        h = self.embedding(x)    # [B, T, H]
        pos_encoding = self.pos_encoder(x)
        h_pos = h + pos_encoding
        mask = (x != 0)
        
        for block in self.encoder_blocks:
            h = block(h_pos, h_pos, h, pad_mask=mask)
        h = h * mask[:, :, None]

        slots = self.pool(h, mask)
        slots = slots - slots.mean(dim=1, keepdim=True)
        slots = F.layer_norm(slots, slots.shape[-1:])
        z_input = slots.reshape(B, -1)
        mu = self.fc_mu(z_input)
        logvar = self.fc_logvar(z_input)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
   
    def decode(self, z, x=None, mode='eval', sos_id=1):
        B, _ = z.shape
        pred_len = F.softplus(self.fc_lenght(z.detach())).squeeze(-1)
        pred_len_i = torch.round(pred_len).long()
        pred_len_i = torch.clamp(pred_len_i, min=1)
        
        z = self.fc_z2h(z)
        slots = z.view(B, self.num_slots, self.hidden_size)
        slots = slots - slots.mean(dim=1, keepdim=True)
        slots = F.layer_norm(slots, slots.shape[-1:])
        max_len_batch = int(pred_len.max().item())
        t = torch.arange(max_len_batch, device=z.device)[None, :] 

        pos_q = self.pos_encoder(torch.empty(B, max_len_batch, self.hidden_size, device=z.device))
        pos_q = pos_q.expand(B, -1, -1).detach().clone()
        pos_q[:, 0, :] = self.embedding.weight[sos_id]
        
        h = self.cross_block(pos_q, slots, slots)  
        h = F.layer_norm(h, h.shape[-1:]) 

        dec_mask = t < pred_len_i[:, None]

        for block in self.decoder_blocks:
            h = block(h, h, h, pad_mask=dec_mask)
        
        logits = self.fc_output(h)
        
        return logits, pred_len
        
    def forward(self, x, mode='eval'):  
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if mode == 'train':
            logits, pred_lenght = self.decode(z, x, mode='train')
        else:
            logits, pred_lenght = self.decode(z, mode='eval')
        
        return logits, mu, logvar, pred_lenght        

def vae_loss(logits, x, mu, logvar, pred_len, beta=0.01, alpha=0.1, pad_id=0):
    B, T_pred, V = logits.shape
    T_x = x.size(1)
    true_len = (x != pad_id).sum(dim=-1)                # [B]
    pred_len_i = torch.round(pred_len).long().clamp(min=1)  # [B]

    t_pred = torch.arange(T_pred, device=x.device)[None, :]   # [1, T_pred]
    t_x = torch.arange(T_x, device=x.device)[None, :]         # [1, T_x]

    pred_mask = t_pred < pred_len_i[:, None]           # [B, T_pred]
    true_mask = t_x < true_len[:, None]                # [B, T_x]

    min_T = min(T_pred, T_x)
    pred_mask = pred_mask[:, :min_T]
    true_mask = true_mask[:, :min_T]

    mask = pred_mask & true_mask                        # [B, T]

    logits = logits[:, :min_T, :].reshape(-1, V)
    targets = x[:, :min_T].reshape(-1)
    mask = mask.reshape(-1)

    rec_loss = F.cross_entropy(logits[mask],targets[mask])
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    len_loss = F.mse_loss(pred_len, true_len.float())
    loss = rec_loss + beta * kl_loss + alpha * len_loss
    return loss, rec_loss, kl_loss, len_loss