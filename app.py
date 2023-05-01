# Importing dependencies
import streamlit as st
import requests
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import GPT2TokenizerFast
from PIL import Image
import os
import json
import argparse
from model import SceneScript

# Set page title, icon and layout
st.set_page_config(page_title="Scene Script", page_icon=":memo:", layout="wide")

# Set page title and description
st.title("Scene Script")
st.markdown("This is a demo of the Scene Script model. The model generates a caption for a given image.")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading model config
model_param = '97M'
with open(f'config/{model_param}.json'.format(model_param)) as f:
    params = json.load(f)

# Encoder Parameters
encoder_params = params.pop('encoder')
# Decoder Parameters
decoder_params = params.pop('decoder')

# Define the data preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((encoder_params['image_size'], encoder_params['image_size'])),
    transforms.ToTensor()
])

# Load the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.add_special_tokens({'bos_token': '<|startoftext|>'})

# Define function for generating caption
def generate_caption(model, image_path, tokenizer):
    # Open image and apply transformations
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Generate caption
    with torch.no_grad():
        encoded = model.encoder(image, return_embeddings=True)
        caption_tokens = [tokenizer.bos_token_id]
        for i in range(decoder_params['max_seq_len']):
            caption_tensor = torch.tensor(caption_tokens).unsqueeze(0).to(device)
            output = model.decoder(caption_tensor, context=encoded)
            last_token_logits = output[0, -1, :]
            next_token_id = torch.argmax(last_token_logits).item()
            caption_tokens.append(next_token_id)
    
    # Decode caption
    caption = tokenizer.decode(caption_tokens, skip_special_tokens=True)
    return caption

# Load the model
model = SceneScript(encoder_params, decoder_params).to(device)
model.load_state_dict(torch.load('weights/scene-script.pth', map_location=device))
model.eval()

# Create sidebar for file upload or image URL
st.sidebar.title("Upload Image")
option = st.sidebar.selectbox('Select Input Option', ['Upload Image', 'Input URL'])

if option == 'Upload Image':
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption='Uploaded Image.', use_column_width=True)
else:
    image_url = st.sidebar.text_input("Enter Image URL")
    if image_url != "":
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.sidebar.image(image, caption='Input Image.', use_column_width=True)
        except:
            st.sidebar.warning("Invalid URL entered. Please try again.")

# Generate caption when 'Generate Caption' button is clicked
if st.button("Generate Caption"):
    if option == 'Upload Image':
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        else:
            caption = generate_caption(model, uploaded_file, tokenizer)
            st.write('\nCaption: ' + caption)
    else:
        if image_url == "":
            st.warning("Please enter an image URL first.")
        else:
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                image_tensor = transform(image).unsqueeze(0)
                caption = generate_caption(model, image_tensor, tokenizer)
                st.success("Generated Caption: " + caption)
            except:
                st.warning("Invalid URL entered. Please try again.")
