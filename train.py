import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from tqdm import tqdm
import json
from tokenizers import Tokenizer
from datetime import datetime
import gc

class GPTConfig:
    def __init__(
        self,
        vocab_size=22588,
        n_embd=768,      # Reduced from 2048
        n_head=12,       # Reduced from 16
        n_layer=8,       # Reduced from 12
        dropout=0.1,
        block_size=256,  # Reduced from 512
        learning_rate=3e-4,
        max_epochs=50,
        batch_size=8,    # Reduced from 64
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
        self.batch_size = batch_size
        self.grad_clip = grad_clip

# Model Architecture
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
        
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss


class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):  # Reduced max_length
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("Tokenizing texts...")
        self.examples = []
        
        for text in tqdm(texts):
            tokens = self.tokenizer.encode(text).ids
            for i in range(0, len(tokens) - max_length, max_length // 2):
                chunk = tokens[i:i + max_length]
                if len(chunk) < max_length:
                    chunk = chunk + [0] * (max_length - len(chunk))
                self.examples.append(chunk)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])

def train():
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Loading Wikipedia data...")
    with open('az_wiki_data.json', 'r', encoding='utf-8') as f:
        wiki_data = json.load(f)
    
    texts = [page['text'] for page in wiki_data.values()]
    tokenizer = Tokenizer.from_file("az_tokenizer.json")
    
    dataset = WikiTextDataset(texts, tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    config = GPTConfig()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,  # Reduced from 4
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,  # Reduced from 4
        pin_memory=True
    )
    
    model = GPT(config)
    model = model.to('cuda')
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    scaler = torch.amp.GradScaler()  # Updated deprecation warning
    
    def run_epoch(split, epoch_num=0):
        is_train = split == 'train'
        model.train(is_train)
        if not is_train:
            model.eval()
            
        loader = train_loader if is_train else val_loader
        losses = []
        
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        
        for it, (x, y) in pbar:
            # Clear memory
            torch.cuda.empty_cache()
            
            x = x.to('cuda', non_blocking=True)
            y = y.to('cuda', non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda'):  # Updated deprecation warning
                logits, loss = model(x, y)
            
            losses.append(loss.item())
            
            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                pbar.set_description(f"epoch {epoch_num+1} iter {it}: train loss {loss.item():.5f}")
                
            # Delete unnecessary tensors
            del x, y, logits
            if is_train:
                del loss
        
        mean_loss = torch.tensor(losses).mean().item()
        return mean_loss
    
    best_val_loss = float('inf')
    
    try:
        for epoch in range(config.max_epochs):
            print(f"\nEpoch {epoch+1}/{config.max_epochs}")
            
            train_loss = run_epoch('train', epoch_num=epoch)
            
            with torch.no_grad():
                val_loss = run_epoch('val')
                
            scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving best model with val_loss: {val_loss:.4f}")
                torch.save(model.state_dict(), 'best_model.pt')
            
            print(f"Epoch {epoch+1}: train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
            
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f'checkpoint_epoch_{epoch+1}.pt')
                
    except KeyboardInterrupt:
        print('Training interrupted, saving checkpoint...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, 'interrupt_checkpoint.pt')

if __name__ == '__main__':
    train()