import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import json
import os
from pathlib  import Path
import lightning.pytorch as pl
from tqdm import tqdm
import urllib
import zipfile
import wget
from transformers import AutoTokenizer

LLAMA_CONTEXT_LENGTH = 2048
IMAGE_CONTEXT_LENGTH = 196
TEXT_CONTEXT_LENGTH = LLAMA_CONTEXT_LENGTH - IMAGE_CONTEXT_LENGTH
IGNORE_INDEX = -100


class SciCapDataset(Dataset):
    def __init__(self, root_dir, split, subfolder="SciCap-No-Subfig-Img"):
        # root_dir is the path to the extracted SCICAP archive 
        self.root_dir = root_dir
        self.split = split
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.image_names = [os.path.abspath(os.path.join(root_dir, subfolder, split, f)) for f in os.listdir(os.path.join(root_dir, subfolder, split))]

        model = "PY007/TinyLlama-1.1B-Chat-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(model) 
        self.tokenizer.padding_side = "right"

    def __getitem__(self, index):
        # Get the image name and caption for the given index
        image_name = self.image_names[index]

        # Load the image from the file
        image = Image.open(image_name)
        inputs = self.processor(images=image, return_tensors="pt")

        caption_fn = image_name.replace("SciCap-No-Subfig-Img", "SciCap-Caption-All").replace(".png", ".json")
        caption = json.load(open(caption_fn))["0-originally-extracted"]
        caption = f' --> This can be described as follows: \n{caption}</s>'  # make it a prompt, assuming image features will be prepended.

        image_tensor = inputs['pixel_values'].squeeze(0)  # (1, 3, 224, 224) -> (3, 224, 224)
        text_token_ids = self.tokenizer.encode(caption, pad_to_max_length=True, max_length=TEXT_CONTEXT_LENGTH + 1).ids  # list of ints

        input_text_ids = text_token_ids[:-1]  # image feature + input_text_ids --> input of autoregressive model
        target_text_ids = [IGNORE_INDEX] * IMAGE_CONTEXT_LENGTH + text_token_ids[1:]  # --> target of autoregressive model

        return {"image_tensor" : image_tensor,
                "input_text_ids": input_text_ids,
                "target_text_ids": target_text_ids,}

    def __len__(self):
        # Return the size of the dataset
        return len(self.image_names)

class SciCapDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, num_workers=0):
        super().__init__()

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        os.makedirs(self.root_dir, exist_ok=True)
        zip_path = os.path.join(self.root_dir, 'scicap_data.zip')
        url = 'https://www.dropbox.com/s/t1sjqesl0pynaxo/scicap_data.zip?dl=1'

        if not os.path.exists(zip_path):
            print(f'Downloading dataset from {url}...')
            wget.download(url, zip_path)

        if not os.path.isdir(os.path.join(self.root_dir, 'SciCap-No-Subfig-Img')):
            print(f'Extracting dataset to {self.root_dir}...')
            with zipfile.ZipFile(zip_path, 'r') as f:
                f.extractall(self.root_dir)
        else:
            print(f'Dataset is already extracted at {os.path.join(self.root_dir, "SciCap-No-Subfig-Img")}...')

            print(f'Dataset is ready at {self.root_dir}!')

    def setup(self, stage=None):
        data_dir = os.path.join(self.root_dir, "scicap_data")

        # Split the data into training and validation sets
        if stage == 'fit' or stage is None:
            self.train_dataset = SciCapDataset(data_dir, 'train')
            self.val_dataset = SciCapDataset(data_dir, 'val')

        # Assign the test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = SciCapDataset(data_dir, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)