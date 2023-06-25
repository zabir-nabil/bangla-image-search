import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
import config as CFG
import cv2

class CLIPModel(nn.Module):
    """CLIP model for Bangla"""
    def __init__(self):
        super(CLIPModel, self).__init__()
        self.image_encoder = models.efficientnet_b2(weights = "EfficientNet_B2_Weights.DEFAULT")
        self.image_encoder.fc = nn.Identity()
        
        self.image_out = nn.Sequential(
            nn.Linear(CFG.image_embedding, 256), nn.ReLU(), nn.Linear(256, 256)
        )

        self.text_encoder = AutoModel.from_pretrained(CFG.text_encoder_model)
        self.target_token_idx = 0


        self.text_out = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 256)
        )
      

    def forward(self, image, text, mask):
        image_vec = self.image_encoder(image)        
        image_vec = self.image_out(image_vec)

        text_out = self.text_encoder(text, mask)
        last_hidden_states = text_out.last_hidden_state

        last_hidden_states = last_hidden_states[:,self.target_token_idx,:]
        text_vec = self.text_out(last_hidden_states.view(-1,768))

        return image_vec, text_vec
    
    def get_image_embeddings(self, image):
        image_vec = self.image_encoder(image)        
        image_vec = self.image_out(image_vec)

        return image_vec
    
    def get_text_embeddings(self, text, mask):
        text_out = self.text_encoder(text, mask)
        last_hidden_states = text_out.last_hidden_state

        last_hidden_states = last_hidden_states[:,self.target_token_idx,:]
        text_vec = self.text_out(last_hidden_states.view(-1,768))

        return text_vec

    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = torch.randn(40, 3, 224, 224).to(device)
    input_ids = torch.randint(5, 300, size=(40, 200)).to(device)
    attention_mask = torch.ones(40, 200).to(device)

    print("Building CLIP")
    clip_model = CLIPModel().to(device)
    print(clip_model)

    img_vec, text_vec = clip_model(images, input_ids, attention_mask)
    print(img_vec.shape)
    print(text_vec.shape)