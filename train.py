import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from model import SiameseResNet
from dataset import VeinDataset
from torchvision import transforms
import os
import random
import numpy as np

# 시드 값을 설정합니다.
seed_value = 777

random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 200
learning_rate = 0.001
batch_size = 32
margin = 0.2  # You can adjust this value

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = VeinDataset(root_dir='../FingerVein3.5/finger_split_2', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = SiameseResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (img1, img2, label) in enumerate(train_loader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        output1, output2 = model(img1, img2)
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))

        optimizer.zero_grad()
        loss_contrastive.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_contrastive.item()}')

# Save model
torch.save(model.state_dict(), './siamese_resnet_2000_02.pth')
