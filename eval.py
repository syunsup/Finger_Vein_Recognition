import torch
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from model import SiameseResNet
from dataset import VeinDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import scipy
import random

# 시드 값을 설정합니다.
seed_value = 777

random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = SiameseResNet().to(device)
model.load_state_dict(torch.load('./siamese_resnet.pth'))

# Prepare test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = VeinDataset(root_dir='../FingerVein3.5/data_val', transform=transform)
#test_dataset = VeinDataset(root_dir='../FingerVein3.5/finger_split_2', transform=transform)
test_loader = DataLoader(dataset=test_dataset, shuffle=False)

model.eval()

scores, truths = [], []

with torch.no_grad():
    for i, (img1, img2, label) in enumerate(test_loader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)
        output1, output2 = model(img1, img2)
        euclidean_distance = F.pairwise_distance(output1, output2)
        scores.append(1. / (euclidean_distance.item() + 1e-6))  # Adding a small constant to avoid division by zero
        truths.append(label.item()) # Assuming that label is 1 for a pair of images from the same class and 0 for a pair of images from different classes.

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(truths, scores)

# Compute EER
eer = scipy.optimize.brentq(lambda x : 1. - x - scipy.interpolate.interp1d(fpr, tpr)(x), 0., 1.)

# Compute TAR at 0.01% FAR
frr = 1 - tpr
far_index = np.where(fpr <= 0.0001)
if far_index[0].size > 0: # Check if there is any threshold with FAR <= 0.01%
    far_index = far_index[0][-1]
    tar_at_far_001 = 1 - frr[far_index]
else:
    tar_at_far_001 = None # Set to None if there is no such threshold

print(f"EER: {eer}, TAR at 0.01% FAR: {tar_at_far_001}")

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_score(truths, scores))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
