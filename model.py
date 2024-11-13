import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer
import requests
from bs4 import BeautifulSoup
import wikipediaapi

class WikiDataset(Dataset):
    def __init__(self, texts, seq_length, tokenizer):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.texts = texts
        
        # Tokenize all texts
        self.encoded = []
        for text in texts:
            self.encoded.extend(self.tokenizer.encode(text))
            
    def __len__(self):
        return len(self.encoded) - self.seq_length
        
    def __getitem__(self, idx):
        chunk = self.encoded[idx:idx + self.seq_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.n_embd // config.n_heads
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        if mask is not None:
            att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class GPTConfig:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout, block_size):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_heads = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.block_size = block_size

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # forward the GPT model
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        mask = torch.ones(b, 1, t, t, device=device)
        mask = torch.tril(mask)
        
        for block in self.transformer.h:
            x = block(x, mask)
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

def get_az_wiki_texts():
    """
    Function to get Azerbaijani Wikipedia articles
    Returns a list of text strings
    """
    wiki = wikipediaapi.Wikipedia('az')
    # This is a placeholder - you'll need to implement proper crawling logic
    # for Azerbaijani Wikipedia articles
    texts = []
    return texts

def train_model():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Get Azerbaijani Wikipedia texts
    texts = get_az_wiki_texts()
    
    # Create dataset
    dataset = WikiDataset(texts, seq_length=128, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=6,
        dropout=0.1,
        block_size=128
    )
    model = GPT(config)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Training loop
    for epoch in range(10):
        for batch_idx, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item():.4f}')
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    model.eval()
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model(context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
            
            if next_token.item() == tokenizer.sep_token_id:
                break
    
    return tokenizer.decode(context[0].tolist())