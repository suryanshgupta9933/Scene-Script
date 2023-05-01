print('Starting Scene Script...')

# Importing the libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import GPT2TokenizerFast
from PIL import Image
import os
import json
import argparse
from tqdm import tqdm
from model import SceneScript

# Take arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, help='Path to Image')
parser.add_argument('--model_config', type=str, default='97M', help='Model Parameters Configuration')
args = parser.parse_args()

# Checking if path is valid
try:
    if args.image_path is None:
        raise ValueError("Enter path to image")
    if not os.path.isfile(args.image_path):
        raise ValueError("Invalid path to image")
except ValueError as e:
    print("Error:", str(e))
    exit()

# Defining Image Path
image_path = args.image_path
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

# Define the data preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((encoder_params['image_size'], encoder_params['image_size'])),
    transforms.ToTensor()
])

# Load the image
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)
print('Image: ' + '\u2713')

# Load the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.add_special_tokens({'bos_token': '<|startoftext|>'})
print('Tokenizer: ' + '\u2713')

# Load the model
print('Initializing Model...')
model = SceneScript(encoder_params, decoder_params).to(device)
print('Model: ' + '\u2713')

# Load the trained weights
model.load_state_dict(torch.load('weights/epoch_20.pth', map_location=device))
model.eval()
print('Loaded weights: ' + '\u2713')

# Inference function
print('Generating Caption...')
def generate_caption(model, image_path, tokenizer):
    # Open image and apply transformations
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Generate caption
    with torch.no_grad():
        encoded = model.encoder(image, return_embeddings=True)
        caption_tokens = [tokenizer.bos_token_id]
        for i in tqdm(range(decoder_params['max_seq_len'])):
            caption_tensor = torch.tensor(caption_tokens).unsqueeze(0).to(device)
            output = model.decoder(caption_tensor, context=encoded)
            last_token_logits = output[0, -1, :]
            next_token_id = torch.argmax(last_token_logits).item()
            caption_tokens.append(next_token_id)
            if next_token_id == tokenizer.eos_token_id:
                break

    # Convert tokens to string
    caption = tokenizer.decode(caption_tokens, skip_special_tokens=True)
    return caption

# Generate caption
caption = generate_caption(model, image_path, tokenizer)
print('\nCaption: ' + caption)