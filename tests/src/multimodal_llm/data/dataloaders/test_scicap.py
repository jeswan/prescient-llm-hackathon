import pytest
import torch
from src.multimodal_llm.data.dataloaders.scicap import SciCapDataset # Assuming the code snippet is in test_sample.py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm 

# Define some image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize the image to 224x224
    transforms.ToTensor(), # Convert the image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Normalize the image with mean and std
])

# Define some test cases with different arguments
test_cases = [
    ("/teamspace/studios/this_studio/data/scicap_data", "train", 106834), # No transformation, expected length is 416000
    ("/teamspace/studios/this_studio/data/scicap_data", "val", 13354), # With transformation, expected length is 416000
    ("/teamspace/studios/this_studio/data/scicap_data", "test", 13355), # With transformation, expected length is 416000
]

# Define a test function with parametrized arguments
@pytest.mark.parametrize("root_dir, split, expected_len", test_cases)
def test_scicap_dataset(root_dir, split, expected_len):
    # Create an instance of the SciCapDataset class
    scicap_dataset = SciCapDataset(root_dir=root_dir, split=split)
    # Assert that the length of the dataset matches the expected value
    assert len(scicap_dataset) == expected_len
    # If the length is not zero, assert that the images and captions are of the correct type and shape
    if expected_len > 0:
        # Get the first image and caption from the dataset
        x = scicap_dataset[0]
        image = x["image"]
        caption = x["text"]

        # Assert that the image is a torch.Tensor
        assert isinstance(image, torch.Tensor)
        # Assert that the image shape is (3, 224, 224)
        # assert image.shape == (3, 224, 224)
        # Assert that the caption is a str
        assert isinstance(caption, str)

@pytest.mark.parametrize("root_dir, split, expected_len", test_cases)
def test_scicap_dataloader(root_dir, split, expected_len):
    scicap_dataset = SciCapDataset(root_dir=root_dir, split=split)
    scicap_dataloader = DataLoader(scicap_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    for batch in tqdm(scicap_dataloader):
        breakpoint()

