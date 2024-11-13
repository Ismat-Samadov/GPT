# Azerbaijani Language GPT Model

This repository contains an implementation of a GPT (Generative Pre-trained Transformer) model trained on Azerbaijani Wikipedia data. The model is designed to understand and generate Azerbaijani text.

## Project Structure
```
.
├── README.md
├── az_tokenizer.json        # Trained tokenizer for Azerbaijani text
├── az_wiki_data.json        # Collected Wikipedia data
├── best_model.pt            # Saved state of the best trained model
├── collect_data.py          # Script for collecting Wikipedia articles
├── generate.py              # Text generation script using the trained model
├── prepare_data.py          # Data preprocessing and tokenizer training
├── push_to_hf.py            # Script to upload the trained model to Hugging Face Model Hub
├── requirements.txt         # Project dependencies
└── train.py                 # GPT model training script
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies based on your system:

For Mac with Apple Silicon (M1/M2):
```bash
# Install PyTorch for Apple Silicon
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Install other required packages
pip install transformers wikipedia-api beautifulsoup4 requests huggingface_hub
```

For other systems:
```bash
pip install -r requirements.txt
```

## Platform-Specific Notes

### Apple Silicon (M1/M2) Macs
- Uses MPS (Metal Performance Shaders) for acceleration
- Optimized memory management for Apple Silicon
- May require specific PyTorch nightly builds

### CUDA-enabled GPUs
- Automatically utilizes CUDA if available
- Implements mixed precision training
- Memory optimization through gradient accumulation

## Data Collection

1. Collect Azerbaijani Wikipedia articles:
```bash
python collect_data.py
```
This will save articles to `az_wiki_data.json`

2. Prepare data and train tokenizer:
```bash
python prepare_data.py
```
This will create `az_tokenizer.json`

## Training

Train the GPT model:
```bash
python train.py
```

The training script:
- Uses mixed precision training
- Implements gradient accumulation
- Saves model checkpoints every 5 epochs
- Saves the best model based on validation loss

## Model Architecture

- Transformer-based architecture
- Configuration adjustable in `train.py`:
  - Embedding dimension: 512
  - Attention heads: 8
  - Layers: 6
  - Block size: 128
  - Batch size: 4

## Text Generation

Generate text using the trained model:
```bash
python generate.py
```
The `generate.py` script:
- Loads the trained model and tokenizer
- Generates text based on a user-provided prompt
- Implements sampling strategies such as nucleus sampling and temperature scaling

## Upload to Hugging Face Model Hub

Upload your trained model to the Hugging Face Model Hub:
```bash
python push_to_hf.py
```
The `push_to_hf.py` script:
- Authenticates with your Hugging Face account
- Creates a new repository for your model (if needed)
- Uploads the trained model, tokenizer, and any other relevant files

## Files Description

- `collect_data.py`: Collects articles from Azerbaijani Wikipedia using categories like history, culture, literature, and geography
- `prepare_data.py`: Preprocesses text and trains a BPE tokenizer
- `train.py`: Contains GPT model implementation and training loop
- `generate.py`: Generates text using the trained model and sampling strategies
- `push_to_hf.py`: Script for uploading the trained model to Hugging Face's Model Hub
- `az_wiki_data.json`: Collected and preprocessed Wikipedia articles
- `az_tokenizer.json`: Trained BPE tokenizer for Azerbaijani text
- `best_model.pt`: Saved state of the best model during training

## Training Output

The model saves:
- Best model state as `best_model.pt`
- Regular checkpoints as `checkpoint_epoch_N.pt`
- Interrupted training state as `interrupt_checkpoint.pt`

## Memory Requirements

- Recommended: GPU with at least 8GB memory
- For larger models: Use gradient accumulation steps
- Adjustable batch size and model size based on available memory

## Troubleshooting

Common Issues:
1. Memory Errors:
   - Reduce batch size
   - Enable gradient accumulation
   - Reduce model size
   - Clear GPU cache regularly

2. PyTorch Installation:
   - For Apple Silicon: Use the nightly build command
   - For CUDA: Install appropriate CUDA version

3. Data Loading:
   - Reduce number of workers if getting process errors
   - Enable pin memory for faster data transfer

## Future Improvements

- [ ] Implement model evaluation metrics
- [ ] Add data augmentation techniques
- [ ] Implement distributed training
- [ ] Add model compression techniques