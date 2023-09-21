import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import json
import os
from pathlib  import Path

# Define a custom Dataset class
class SciCapDataset(Dataset):
    def __init__(self, root_dir, split, subfolder="SciCap-No-Subfig-Img"):
        # root_dir is the path to the extracted SCICAP archive 
        self.root_dir = root_dir
        self.split = split

        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

        

        self.image_names = [os.path.abspath(os.path.join(root_dir, subfolder, split, f)) for f in os.listdir(os.path.join(root_dir, subfolder, split))]
        self.captions = {}


        for filename in self.image_names:
            caption_fn = filename.replace(subfolder, "SciCap-Caption-All").replace(".png", ".json")
            self.captions[Path(filename).stem] = json.load(open(caption_fn))["0-originally-extracted"]

    def __getitem__(self, index):
        # Get the image name and caption for the given index
        image_name = self.image_names[index]
        caption = self.captions[Path(image_name).stem]

        # Load the image from the file
        image = Image.open(image_name)
        inputs = self.processor(images=image, return_tensors="pt")
        

        return {"image": inputs['pixel_values'].squeeze(0), "text": caption}

    def __len__(self):
        # Return the size of the dataset
        return len(self.image_names)

