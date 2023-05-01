import streamlit as st
import torch
from PIL import Image
import requests
from io import BytesIO
from scene_script import generate_caption

st.set_page_config(page_title="Scene Script", page_icon=":memo:", layout="wide")

# Define the data preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.add_special_tokens({'bos_token': ''})

# Load the model
model = torch.load('scene-script.pth', map_location=torch.device('cpu'))
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
if st.sidebar.button("Generate Caption"):
    if option == 'Upload Image':
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        else:
            image_tensor = transform(image).unsqueeze(0)
            caption = generate_caption(model, image_tensor, tokenizer)
            st.success("Generated Caption: " + caption)
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
