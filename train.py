import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import json
from tokenizers import Tokenizer
import wandb
from datetime import datetime

class GPTConfig:
    def __init__(
        self,
        vocab_size=22588,
        n_embd=2048,     # Increased for A100
        n_head=16,       # Increased for A100
        n_layer=12,      # Increased for A100
        dropout=0.1,
        block_size=512,
        learning_rate=3e-4,
        max_epochs=50,
        batch_size=64,   # Increased for A100
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

class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
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

def train_model():
    # Initialize wandb
    wandb.init(project="azerbaijani-gpt")
    
    # Load the collected Wikipedia data
    print("Loading Wikipedia data...")
    with open('az_wiki_data.json', 'r', encoding='utf-8') as f:
        wiki_data = json.load(f)
    
    # Extract texts
    texts = [page['text'] for page in wiki_data.values()]
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("az_tokenizer.json")
    
    # Create dataset
    dataset = WikiTextDataset(texts, tokenizer)
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create config and model
    config = GPTConfig()
    wandb.config.update(config.__dict__)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    model = GPT(config)
    model = model.to('cuda')
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    def run_epoch(split, epoch_num=0):
        is_train = split == 'train'
        model.train(is_train)
        loader = train_loader if is_train else val_loader
        
        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        
        for it, (x, y) in pbar:
            x = x.to('cuda', non_blocking=True)
            y = y.to('cuda', non_blocking=True)
            
            with torch.cuda.amp.autocast():  # Mixed precision
                logits, loss = model(x, y)
            
            losses.append(loss.item())
            
            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch_num,
                })
                
                pbar.set_description(f"epoch {epoch_num+1} iter {it}: train loss {loss.item():.5f}")
        
        mean_loss = torch.tensor(losses).mean().item()
        return mean_loss
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.max_epochs):
        train_loss = run_epoch('train', epoch_num=epoch)
        val_loss = run_epoch('val')
        scheduler.step()
        
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving best model with val_loss: {val_loss:.4f}")
            torch.save(model.state_dict(), 'best_model.pt')
        
        print(f"Epoch {epoch+1}: train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'checkpoint_epoch_{epoch+1}.pt')

if __name__ == '__main__':
    print("Starting training...")
    train_model()