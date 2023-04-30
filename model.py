"""
The model is based on the ViT and Transformer architecture.
The x-transformer library is a python wrapper by lucidrains (Phil Wang)
(https://github.com/lucidrains/x-transformers)
"""

# Importing Libraries
import torch
import torch.nn as nn
from x_transformers import ViTransformerWrapper, TransformerWrapper, Encoder, Decoder

# Define the model architecture
class SceneScript(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(SceneScript, self).__init__()
        
        # Encoder parameters
        encoder_attn_layers_params = encoder_params.pop('attn_layers')
        encoder_attn_layers = Encoder(**encoder_attn_layers_params)
        encoder_params['attn_layers'] = encoder_attn_layers

        # Decoder parameters
        decoder_attn_layers_params = decoder_params.pop('attn_layers')
        decoder_attn_layers = Decoder(**decoder_attn_layers_params)
        decoder_params['attn_layers'] = decoder_attn_layers

        # Define the image encoder
        self.encoder = ViTransformerWrapper(**encoder_params)

        # Define the caption decoder
        self.decoder = TransformerWrapper(**decoder_params)
        
    def forward(self, img, caption):
        # Encode the image
        encoded = self.encoder(img, return_embeddings=True)
        # Decode the caption
        output = self.decoder(caption, context=encoded)
        
        return output