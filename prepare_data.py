import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import numpy as np
from tqdm import tqdm

class AzerbaijaniTokenizer:
    def __init__(self, vocab_size=50000):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
        ])
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Punctuation(),
        ])
        
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            min_frequency=2
        )
    
    def train(self, texts):
        """Train the tokenizer on the given texts"""
        print("Training tokenizer...")
        self.tokenizer.train_from_iterator(texts, trainer=self.trainer)
        
    def save(self, path):
        """Save the tokenizer to a file"""
        self.tokenizer.save(path)
        
    def load(self, path):
        """Load the tokenizer from a file"""
        self.tokenizer = Tokenizer.from_file(path)
        
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print("Tokenizing texts...")
        self.examples = []
        
        for text in tqdm(texts):
            # Tokenize the text
            tokens = self.tokenizer.encode(text).ids
            
            # Create sequences of max_length tokens
            for i in range(0, len(tokens) - max_length, max_length // 2):
                chunk = tokens[i:i + max_length]
                if len(chunk) < max_length:
                    # Pad if necessary
                    chunk = chunk + [0] * (max_length - len(chunk))
                self.examples.append(chunk)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Return input and target sequences (for next token prediction)
        tokens = self.examples[idx]
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])

def prepare_data_and_tokenizer():
    # Load the collected Wikipedia data
    print("Loading Wikipedia data...")
    with open('az_wiki_data.json', 'r', encoding='utf-8') as f:
        wiki_data = json.load(f)
    
    # Extract texts
    texts = [page['text'] for page in wiki_data.values()]
    
    # Create and train tokenizer
    tokenizer = AzerbaijaniTokenizer(vocab_size=50000)
    tokenizer.train(texts)
    
    # Save the tokenizer
    tokenizer.save("az_tokenizer.json")
    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Create dataset
    dataset = WikiTextDataset(texts, tokenizer.tokenizer)
    
    # Create data loaders
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Total sequences: {len(dataset)}")
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    
    return tokenizer, train_loader, val_loader

if __name__ == "__main__":
    tokenizer, train_loader, val_loader = prepare_data_and_tokenizer()