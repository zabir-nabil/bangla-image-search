import torch

debug = True
image_path = ""
captions_path = ""

dataset_root = "/data/comps"
train_json = "/data/comps/train.json"
val_json = "/data/comps/val.json"


batch_size = 200
num_workers = 5
head_lr = 1e-3
image_encoder_lr = 1e-4
text_encoder_lr = 1e-5
weight_decay = 1e-3
patience = 2
factor = 0.8
epochs = 350

gpu = 1
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

# model_name = 'efficientnet_b2'
# image_embedding = 1408

model_name = 'resnet50'
image_embedding = 1000

text_encoder_model = "neuralspace-reverie/indic-transformers-bn-bert" 
text_tokenizer = "neuralspace-reverie/indic-transformers-bn-bert"
max_length = 100

model_tag = f"{model_name}_{text_encoder_model.replace('/', '_')}_aug"
log_tag = model_tag

pretrained = True # for both image encoder and text encoder
trainable = True # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# proj. head
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1