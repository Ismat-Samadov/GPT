import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import json
from tokenizers import Tokenizer
import wandb  # for tracking training progress
import os
from datetime import datetime

# Configuration class
class GPTConfig:
    def __init__(
        self,
        vocab_size=22588,
        n_embd=768,
        n_head=12,
        n_layer=6,
        dropout=0.1,
        block_size=511,  # sequence length - 1 (for next token prediction)
        learning_rate=3e-4,
        max_epochs=50,
        warmup_tokens=512*20,
        final_tokens=2*len(open('az_wiki_data.json', 'r').read()),
        num_workers=4,
        batch_size=32,
        grad_clip=1.0,
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.block_size = block_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.warmup_tokens = warmup_tokens
        self.final_tokens = final_tokens
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.grad_clip = grad_clip

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.w_k = nn.Linear(config.n_embd, config.n_embd)
        self.w_q = nn.Linear(config.n_embd, config.n_embd)
        self.w_v = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
    def forward(self, x):
        B, T, C = x.size()
        k = self.w_k(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.w_q(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.block_size = config.block_size
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        # forward the GPT model
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

def train_model():
    # Initialize wandb
    wandb.init(project="azerbaijani-gpt")
    
    # Load config
    config = GPTConfig()
    wandb.config.update(config.__dict__)
    
    # Create model
    model = GPT(config)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Load tokenizer and create dataloaders
    tokenizer = Tokenizer.from_file("az_tokenizer.json")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    def run_epoch(split, epoch_num=0):
        is_train = split == 'train'
        model.train(is_train)
        
        # Load the appropriate dataloader
        if is_train:
            loader = train_loader
        else:
            loader = val_loader
            
        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        for it, (x, y) in pbar:
            
            # place data on the correct device
            x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
            y = y.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # forward the model
            with torch.set_grad_enabled(is_train):
                logits, loss = model(x, y)
                losses.append(loss.item())
                
            if is_train:
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                
                # report progress
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': config.learning_rate,
                    'epoch': epoch_num,
                })
                
                pbar.set_description(f"epoch {epoch_num+1} iter {it}: train loss {loss.item():.5f}")
        
        mean_loss = torch.tensor(losses).mean().item()
        return mean_loss
    
    # training loop
    best_val_loss = float('inf')
    for epoch in range(config.max_epochs):
        train_loss = run_epoch('train', epoch_num=epoch)
        val_loss = run_epoch('val')
        
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving best model with val_loss: {val_loss:.4f}")
            torch.save(model.state_dict(), 'best_model.pt')
            
        print(f"Epoch {epoch+1}: train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

if __name__ == '__main__':
    # First install wandb if not already installed
    try:
        import wandb
    except ImportError:
        print("Installing wandb...")
        os.system('pip install wandb')
        import wandb
    
    print("Starting training...")
    train_model()