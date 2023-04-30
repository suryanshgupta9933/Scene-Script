"""
The model is based on the ViT and Transformer architecture.
The x-transformer library is a python wrapper by lucidrains (Phil Wang)
(https://github.com/lucidrains/x-transformers)
"""

# Importing Libraries
import torch
import torch.nn as nn
from x_transformers import ViTransformerWrapper, TransformerWrapper, Encoder, Decoder

# 

class SceneScript(nn.Module):
    def __init__(self):
        super(SceneScript, self).__init__()
        # Define the image encoder
        self.encoder = ViTransformerWrapper(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            attn_layers=Encoder(
                dim=DIMENSION,
                depth=LAYERS,
                heads=HEADS
            )
        )
        # Define the caption decoder
        self.decoder = TransformerWrapper(
            num_tokens=VOCAB_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            attn_layers=Decoder(
                dim=DIMENSION,
                depth=LAYERS,
                heads=HEADS,
                cross_attend=True
            )
        )
        
    def forward(self, img, caption):
        # Encode the image
        encoded = self.encoder(img, return_embeddings=True)
        # Decode the caption
        output = self.decoder(caption, context=encoded)
        return output