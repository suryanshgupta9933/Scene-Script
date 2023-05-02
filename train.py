print('Starting Scene Script...')

# Importing the libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import GPT2TokenizerFast
from PIL import Image
import os
import json
import argparse
from model import SceneScript

# Take arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str, default='97M', help='Model Parameters Configuration')
parser.add_argument('--images_dir', type=str, help='Path to Images Directory')
parser.add_argument('--caption_file', type=str, help='Path to Captions File')
args = parser.parse_args()

# Checking if paths are valid
try:
    if args.images_dir is None:
        raise ValueError("Enter path to images directory")
    if not os.path.isdir(args.images_dir):
        raise ValueError("Invalid path to images directory")
    if args.caption_file is None:
        raise ValueError("Enter path to captions file")
    if not os.path.isfile(args.caption_file):
        raise ValueError("Invalid path to captions file")
except ValueError as e:
    print("Error:", str(e))
    exit()

# Defining Paths
images_dir = args.images_dir
caption_file = args.caption_file
print('Paths: ' + '\u2713')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device set to ' + str(device) + ': \u2713')

# Loading model config
model_param = args.model_config
with open(f'config/{model_param}.json'.format(model_param)) as f:
    params = json.load(f)

# Encoder Parameters
encoder_params = params.pop('encoder')
# Decoder Parameters
decoder_params = params.pop('decoder')
print('Model Config: ' + '\u2713')

# Hyperparameters
train_params = params.pop('train')
print('Hyperparameters: ' + '\u2713')

# Define the data preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((encoder_params['image_size'], encoder_params['image_size'])),
    transforms.ToTensor()
])

# Load the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.add_special_tokens({'bos_token': '<|startoftext|>'})
print('Tokenizer: ' + '\u2713')

# Preprocessing the captions
def preprocess(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) == 2:
            image_file, caption = parts
            caption = caption.replace('"', '')
            new_line = f"{image_file},{caption}\n"
            new_lines.append(new_line)
    
    with open(file_path, 'w') as f:
        f.writelines(new_lines)

preprocess(caption_file)
print('Preprocessing Captions: ' + '\u2713')

# Defining the dataset class
class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, captions_file, tokenizer, transform=None):
        self.img_dir = img_dir
        self.captions = []
        self.transform = transform
        self.tokenizer = tokenizer
        # Load captions from file
        with open(captions_file, 'r') as f:
            for line in f:
                image, caption = line.strip().rsplit(',', 1)
                self.captions.append((image, caption))
                
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        image_file, caption = self.captions[idx]
        # Load image
        image = Image.open(os.path.join(self.img_dir, image_file)).convert('RGB')
        image = self.transform(image)
        # Tokenize caption and convert to tensor
        caption_tokens = self.tokenizer.encode(caption, add_special_tokens=False, truncation=True, max_length=62, padding='max_length')
        caption_tokens = [self.tokenizer.bos_token_id] + caption_tokens + [self.tokenizer.eos_token_id]
        caption_tensor = torch.tensor(caption_tokens).long()
        return image, caption_tensor

# Loading dataset
dataset = Flickr8kDataset(images_dir, caption_file, tokenizer, transform=transform)
print('Dataset: ' + '\u2713')

# Checking Shapes
for i in dataset:
    print('Image Shape: ' + str(i[0].shape))
    print('Caption Shape: ' + str(i[1].shape))
    break
    
# Splitting dataset into train and validation sets
split = 0.9
dataset_size = len(dataset)
train_size = int(dataset_size*split)
val_size = dataset_size - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
print('Train and Validation Split: ' + '\u2713')

# Creating dataloaders
train_loader = DataLoader(train_set, batch_size=train_params['batch_size'], shuffle=True)
val_loader = DataLoader(val_set, batch_size=train_params['batch_size'], shuffle=True)
print('Dataloading into batches of ' + str(train_params['batch_size']) + ': \u2713')

# Initialize the model
print('Initializing Model...')
model = SceneScript(encoder_params, decoder_params).to(device)
print('Model: ' + '\u2713')
print("Model Parameters:", int(sum(p.numel() for p in model.parameters())/1e6), "M")

# Define the loss function
print('Loss Function: ' + '\u2713')
criterion = nn.CrossEntropyLoss()

# Define the optimizer
print('Optimizer: ' + '\u2713')
optimizer = optim.AdamW(model.parameters(), lr=train_params['lr'])

# Define the training loop
print('Training...')
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for epoch in range(train_params['epochs']):
        total_loss = 0
        for i, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            target = captions[:, 1:]
            decoder_input = captions[:, :-1]
            encoded = model.encoder(images, return_embeddings=True)
            output = model.decoder(decoder_input, context=encoded)

            loss = criterion(output.reshape(-1, output.size(2)), target.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i+1) % 500 == 0:
                print(f"Epoch [{epoch+1}/{train_params['epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{train_params['epochs']}], Avg Loss: {avg_loss:.4f}")

# Define the validation loop
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (images, captions) in enumerate(val_loader):
            images = images.to(device)
            captions = captions.to(device)
            target = captions[:, 1:]
            decoder_input = captions[:, :-1]
            encoded = model.encoder(images, return_embeddings=True)
            output = model.decoder(decoder_input, context=encoded)

            loss = criterion(output.reshape(-1, output.size(2)), target.reshape(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")

# Train the model
train(model, train_loader, criterion, optimizer, device)
print('Training Complete: ' + '\u2713')

# Validate the model
validate(model, val_loader, criterion, device)
print('Validation Complete: ' + '\u2713')

# Save the model
os.makedirs('weights', exist_ok=True)
torch.save(model.state_dict(), 'weights/scene-script.pth')
print('Model Saved to weights/scene-script.pth: ' + '\u2713')