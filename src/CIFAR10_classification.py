import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataset_CIFAR10 import PixelClassDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 36
LEARNING_RATE = 0.0005
NUM_CLASSES = 256  # 256 Klassen pro Kanal

# Dataset definition
"""
class PixelClassDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.data = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform

    def __getitem__(self, index):
        img, _ = self.data[index]
        img_tensor = transforms.ToTensor()(img)  # Convert to tensor
        label_tensor = (img_tensor * 255).long()  # Create label tensor
        return img_tensor, label_tensor

    def __len__(self):
        return len(self.data)
"""
# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Dataset and DataLoader
train_dataset = PixelClassDataset(root='./data', train=True, transform=transform)
test_dataset = PixelClassDataset(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model definition
class ConvNet(nn.Module):
    def __init__(self, num_classes=256):
        super(ConvNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv7 = nn.Conv2d(32, num_classes * 3, kernel_size=3, padding=1)  # 3 Kanäle × 256 Klassen

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.upsample1(x)
        x = F.relu(self.conv5(x))
        x = self.upsample2(x)
        x = F.relu(self.conv6(x))
        x = self.upsample3(x)
        x = self.conv7(x)  # Kein Sigmoid, weil Klassifikation
        return x

model = ConvNet(num_classes=NUM_CLASSES).to(device) 

# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

def compute_loss(outputs, targets):
    red_logits, green_logits, blue_logits = torch.chunk(outputs, chunks=3, dim=1) #teilt torch.Size([36, 768, 32, 32]) in 3X torch.Size([36, 256, 32, 32]) auf
    
    red_targets = targets[:, 0, :, :] # weil Rot-Wert dder pixel hier gespeichert
    green_targets = targets[:, 1, :, :] # weil Grün-Wert dder pixel hier gespeichert
    blue_targets = targets[:, 2, :, :] # weil Blau-Wert dder pixel hier gespeichert

    #computing CrossEntropyLoss for all colors
    red_loss = criterion(red_logits, red_targets)
    green_loss = criterion(green_logits, green_targets)
    blue_loss = criterion(blue_logits, blue_targets)

    return (red_loss + green_loss + blue_loss) / 3

# Training and validation loop
#model = ConvNet(num_classes=NUM_CLASSES).to(device)
#optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch(epoch_index, writer):
    running_loss = 0.0
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch_index}"):
        grayscale_images = transforms.functional.rgb_to_grayscale(images).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(grayscale_images)

        loss = compute_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch_index)
    return avg_loss

def validation_loss(epoch_index, writer):
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, (vimages, vlabels) in enumerate(test_loader):
            grayscale_images = transforms.functional.rgb_to_grayscale(vimages).to(device)
            vlabels = vlabels.to(device)
            voutputs = model(grayscale_images)
            vloss = compute_loss(voutputs, vlabels)
            running_vloss += vloss.item()
            
    avg_vloss = running_vloss / len(test_loader)
    writer.add_scalar('Loss/validate', avg_vloss, epoch_index)
    return avg_vloss

# Training the model
def train_model():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/CIFAR10_pixel_classification')

    best_vloss = float('inf')

    for epoch in tqdm(range(EPOCHS),desc="Progress for complete training (percentage of finished EPOCHS)"):
        print('EPOCH {}:'.format(epoch))

        model.train(True)
        avg_loss = train_one_epoch(epoch, writer)
        avg_vloss = validation_loss(epoch, writer)

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = './models_CIFAR10_classification/model_{}_{}'.format( timestamp, epoch)
            torch.save(model.state_dict(), model_path)
            print(f"saved checkpoint in {model_path}")

# Visualization
def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    return image

def plot_images(original, grayscale, colorized):
    original_img = im_convert(original[0])
    grayscale_img = grayscale[0].cpu().numpy()[0]
    colorized_img = im_convert(colorized[0])
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(original_img)
    ax[0].set_title("Original RGB Image")
    ax[0].axis('off')
    ax[1].imshow(grayscale_img, cmap='gray')
    ax[1].set_title("Grayscale Image")
    ax[1].axis('off')
    ax[2].imshow(colorized_img)
    ax[2].set_title("Colorized by Model")
    ax[2].axis('off')
    plt.show()

def plot_examples(model=model):
    model.eval().to(device)
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            grayscale_images = transforms.functional.rgb_to_grayscale(images).to(device)
            images = images.to(device)
            outputs = model(grayscale_images)
            red_logits, green_logits, blue_logits = torch.chunk(outputs, chunks=3, dim=1)
            red_classes = red_logits.argmax(dim=1)
            green_classes = green_logits.argmax(dim=1)
            blue_classes = blue_logits.argmax(dim=1)
            colorized_output = torch.stack([red_classes, green_classes, blue_classes], dim=1)
            plot_images(images, grayscale_images, colorized_output)
            break


