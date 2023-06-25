### search image
from multiprocessing import process
import torch
from transformers import AutoTokenizer
import config as CFG
from CLIP_model import CLIPModel
import cv2
import os
import torch
from glob import glob
import albumentations as A
import torch.nn.functional as F


def load_model(device, model_path):
    """load model and tokenizer"""
    model = CLIPModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
    return model, tokenizer

def process_image(img_path):
    imgs = []
    for ip in img_path:
        transforms_infer = A.Compose(
                [
                    A.Resize(CFG.size, CFG.size, always_apply=True),
                    A.Normalize(max_pixel_value=255.0, always_apply=True),
                ])
        image = cv2.imread(ip)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transforms_infer(image=image)['image']
        image = torch.tensor(image).permute(2, 0, 1).float()
        print(image.shape)
        imgs.append(image)
    imgs = torch.stack(imgs)
    return imgs

def process_text(caption, tokenizer):
    caption = tokenizer(caption, padding = True)
    e_text = torch.Tensor(caption["input_ids"]).long()
    mask = torch.Tensor(caption["attention_mask"]).long()
    return e_text, mask

def search_images(search_text, image_path, k = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(device, model_path="models/clip_bangla.pt")
    image_filenames = glob(image_path + "/*.jpg") + glob(image_path + "/*.JPEG") + glob(image_path + "/*.JPG") + glob(image_path + "/*.png") + glob(image_path + "/*.bmp")
    print(f"Searching in image database >> {image_filenames}")
    imgs = process_image(image_filenames)

    if type(search_text) != list:
        search_text = [search_text]
    e_text, mask = process_text(search_text, tokenizer)

    with torch.no_grad():
        imgs = imgs.to(device)
        e_text = e_text.to(device)
        mask = mask.to(device)
        img_embeddings = model.get_image_embeddings(imgs)
        text_embeddings = model.get_text_embeddings(e_text, mask)
        image_embeddings_n = F.normalize(img_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = text_embeddings_n @ image_embeddings_n.T
        print(dot_similarity.shape)
    top_k_vals, top_k_indices = torch.topk(dot_similarity.detach().cpu(), min(k, len(image_filenames)))
    top_k_vals = top_k_vals.flatten()
    top_k_indices = top_k_indices.flatten()
    print(top_k_indices)
    print(top_k_vals)
    ### log
    images_ret = []
    scores_ret = []
    for i in range(len(top_k_indices)):
        print(f"{image_filenames[int(top_k_indices[i])]} :: {top_k_vals[i]}")
        images_ret.append(image_filenames[int(top_k_indices[i])])
        scores_ret.append(float(top_k_vals[i]))

    return images_ret, scores_ret, top_k_vals, top_k_indices

if __name__ == '__main__':
    img_filenames = ["demo_images/1280px-Cox's_Bazar_Sunset.JPG", "demo_images/Cox's_Bazar,_BangladeshThe_sea_is_calm.jpg", "demo_images/Panta_Vaat_Hilsha_Fisha_VariousVarta_2012.JPG",
                  "demo_images/Pohela_boishakh_10.jpg", "demo_images/Sundarban_Tiger.jpg"]
    captions = ["সমুদ্র সৈকতের তীরে সূর্যাস্ত", "গাঢ় নীল সমুদ্র ও এক রাশি মেঘ", "পান্তা ভাত ইলিশ ও মজার খাবার", "এক দল মানুষ পহেলা বৈশাখে নাগর দোলায় চড়তে এসেছে", "সুন্দরবনের নদীর পাশে একটি বাঘ"]

    search_images("সমুদ্র সৈকতের তীরে সূর্যাস্ত", "demo_images/", k = 10)