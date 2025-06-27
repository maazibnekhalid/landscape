import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer
import numpy as np

# Define TextEncoder and Generator classes (same as your model definitions)
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(x)
        return hidden.squeeze(0)

class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim, feature_maps=64):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim

        self.fc = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(feature_maps * 8 * 4 * 4),
            nn.ReLU(True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, noise, text_embedding):
        x = torch.cat((noise, text_embedding), dim=1)
        x = self.fc(x)
        x = x.view(-1, 512, 4, 4)
        img = self.deconv(x)
        return img

# Initialize device, tokenizer, and models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
embedding_dim = 256
vocab_size = tokenizer.vocab_size

text_encoder = TextEncoder(vocab_size, embedding_dim).to(device)
generator = Generator(noise_dim=100, embed_dim=embedding_dim).to(device)

# Load pretrained model weights
text_encoder.load_state_dict(torch.load('text_encoder_epoch_50.pth', map_location=device))
generator.load_state_dict(torch.load('generator_epoch_50.pth', map_location=device))

text_encoder.eval()
generator.eval()

# Streamlit app
st.title("Text-to-Image Generator")

input_text = st.text_input("Enter a description:", "a beautiful landscape with mountains")

if st.button("Generate Image"):
    with torch.no_grad():
        # Tokenize input text
        encoding = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)

        # Get text embedding
        text_embedding = text_encoder(input_ids)

        # Generate noise vector
        noise = torch.randn(1, 100).to(device)

        # Generate image
        fake_image = generator(noise, text_embedding)
        fake_image = fake_image.squeeze(0)
        fake_image = (fake_image * 0.5) + 0.5  # Denormalize to [0,1]
        fake_image = fake_image.cpu().permute(1, 2, 0).numpy()
        fake_image = np.clip(fake_image, 0, 1)

        # Display image
        st.image(fake_image, caption=input_text, use_column_width=True)