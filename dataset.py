import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
from torchvision import transforms

class VeinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.finger_folders = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]

    def __len__(self):
        return len(self.finger_folders)

    def __getitem__(self, idx):
        finger_folder = self.finger_folders[idx]
        finger_path = os.path.join(self.root_dir, finger_folder)
        images = [f for f in os.listdir(finger_path) if os.path.isfile(os.path.join(finger_path, f))]
        
        positive_image_name = random.choice(images)
        positive_image_path = os.path.join(finger_path, positive_image_name)
        positive_image = Image.open(positive_image_path)
        
        # Select negative image randomly
        if random.random() > 0.5:
            negative_finger_folder = random.choice([d for d in self.finger_folders if d != finger_folder])
            negative_finger_path = os.path.join(self.root_dir, negative_finger_folder)
            negative_images = [f for f in os.listdir(negative_finger_path) if os.path.isfile(os.path.join(negative_finger_path, f))]
            negative_image_name = random.choice(negative_images)
            negative_image_path = os.path.join(negative_finger_path, negative_image_name)
            negative_image = Image.open(negative_image_path)
            label = 0
        else:
            negative_image_name = random.choice([img for img in images if img != positive_image_name])
            negative_image_path = os.path.join(finger_path, negative_image_name)
            negative_image = Image.open(negative_image_path)
            label = 1
        
        if self.transform:
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return positive_image, negative_image, torch.tensor(label, dtype=torch.float32)