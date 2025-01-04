
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import cv2
from torchvision.datasets.utils import download_and_extract_archive
import os
from torchvision.datasets import ImageFolder
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EPOCHS = 30
BATCH_SIZE = 3
LEARNING_RATE = 0.00005
NUM_CLASSES = 256  # 2 Kanäle (a und b)

# Dataset definition
transform = transforms.Compose([
    transforms.Resize((320, 320))#,
    #transforms.ToTensor()
])
resize_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])

# 2. Create a custom dataset classimport os
import tarfile
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from skimage.color import rgb2lab

"""
class PixelClassOpenImages(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.split = 'train' if train else 'val'
        self.transform = transform or transforms.ToTensor()
        self.dataset_url = "https://storage.googleapis.com/openimages/2018_04/train/train_00.tar.gz"
        
        if download:
            self._download_and_extract()
        
        dataset_path = os.path.join(self.root, self.split)
        if not os.path.exists(dataset_path):
            raise RuntimeError(f"Dataset not found at {dataset_path}. Ensure it was downloaded and extracted correctly.")
        
        self.data = ImageFolder(dataset_path)

    def _download_and_extract(self):
        archive_name = f"{self.split}.tar.gz"
        archive_path = os.path.join(self.root, archive_name)
        extracted_path = os.path.join(self.root, self.split)
        
        if not os.path.exists(extracted_path):
            print(f"Downloading Open Images dataset split '{self.split}' to {archive_path}...")
            download_and_extract_archive(self.dataset_url, download_root=self.root, filename=archive_name)
            print("Download and extraction complete.")
        
        if not os.path.exists(extracted_path):
            with tarfile.open(archive_path, 'r:gz') as tar:
                print(f"Extracting {archive_name}...")
                tar.extractall(path=self.root)
                print("Extraction complete.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]  # Load image and its label
        
        # Resize the image
        if self.transform:
            image = self.transform(image)

        # Convert to LAB color space
        image_np = np.array(image.permute(1, 2, 0))  # Convert from CxHxW to HxWxC
        lab_image = rgb2lab(image_np).astype(np.float32)  # Convert to LAB space

        # Normalize LAB channels
        lab_image[..., 0] /= 100.0  # Normalize L channel
        lab_image[..., 1:] = (lab_image[..., 1:] + 128.0)

        # Assign pixel values as labels for all channels
        pixel_labels = torch.tensor(lab_image[..., 1:], dtype=torch.uint8).permute(2, 0, 1).long()

        return torch.tensor(lab_image).permute(2, 0, 1), pixel_labels
"""
class PixelClassImagenette(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.split = 'train' if train else 'val'
        self.transform = transform or transforms.ToTensor()
        self.dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

        if download:
            self._download_and_extract()

        dataset_path = os.path.join(self.root, 'imagenette2-320', self.split)
        self.data = ImageFolder(dataset_path)

    def _download_and_extract(self):
        archive_path = os.path.join(self.root, 'imagenette2-320.tgz')
        extracted_path = os.path.join(self.root, 'imagenette2-320')
        if not os.path.exists(extracted_path):
            print(f"Downloading Imagenette dataset to {archive_path}...")
            download_and_extract_archive(self.dataset_url, download_root=self.root)
            print("Download and extraction complete.")
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]  # Load image and its label
        
        # Resize the image
        if self.transform:
            image = self.transform(image)

        # Convert to LAB color space
        image_np = np.array(image.permute(1, 2, 0))  # Convert from CxHxW to HxWxC
        lab_image = rgb2lab(image_np).astype(np.float32)  # Convert to LAB space

        # Normalize LAB channels (optional, based on your needs)
        lab_image[..., 0] /= 100.0  # Normalize L channel
        lab_image[..., 1:] = (lab_image[..., 1:] + 128.0) #/ 255.0  # Normalize A and B channels

        # Assign pixel values as labels for all channels
        pixel_labels = torch.tensor(lab_image[..., 1:], dtype=torch.uint8).permute(2, 0, 1).long()
        #pixel_labels2 = torch.tensor(lab_image[:,:,1:], dtype=torch.uint8).permute(2, 0, 1).long()
        #pixel_labels2 = torch.from_numpy(lab_image[:,:,1:].astype(int)).permute(2, 0, 1).long()

        #print(pixel_labels.min(), pixel_labels.max())
        #pixel_labels = torch.round(torch.tensor(lab_image[..., 1:] )).astype(np.uint8).permute(2, 0, 1).long()  # Shape [3, H, W]
        #pixel_labels = torch.tensor(lab_image, dtype=torch.uint8).permute(2, 0, 1).long()  # Shape [3, H, W]
        #print(pixel_labels[1].min(), pixel_labels[1].max())
        #print(pixel_labels.shape)
        #print(pixel_labels2.shape)
        #print(image)
        return torch.tensor(lab_image).permute(2, 0, 1), pixel_labels
        


try:
    #train_dataset = PixelClassOpenImages(root='./data_OpenImages_lab', train=True, transform=transform, download=True)
    #val_dataset = PixelClassOpenImages(root='./data_OpenImages_lab', train=False, transform=transform, download=True)
    train_dataset = PixelClassImagenette(root='./data_Imagenette_lab', train=True, transform=resize_transform, download=True)
    test_dataset = PixelClassImagenette(root='./data_Imagenette_lab', train=False, transform=resize_transform, download=True)
except RuntimeError:
    #train_dataset = PixelClassOpenImages(root='./data_OpenImages_lab', train=True, transform=transform, download=False)
    #val_dataset = PixelClassOpenImages(root='./data_OpenImages_lab', train=False, transform=transform, download=False)
    train_dataset = PixelClassImagenette(root='./data', train=True, transform=resize_transform, download=True)
    test_dataset = PixelClassImagenette(root='./data', train=False, transform=resize_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
#train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
#val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)



#function to convert rgb to lab 
def rgb_to_lab(image):
    try:
        image = image.cpu().clone().detach().numpy()
    except AttributeError:
        image = np.array(image) #converting PIL image to array

    image = np.array(image)
    lab_image = rgb2lab(image)
    print(lab_image.shape)
    lab_image = np.transpose(lab_image, (2, 0, 1))
    print(lab_image.shape)
    lab_image = torch.from_numpy(lab_image)
    #lab_image = lab_image.resize_(320,320)
    #lab_image= cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    #lab_image = Image.fromarray(image, mode="LAB")
    #lab_image = Image.fromarray(lab_image)#converting back to PIL Image
    #lab_image = torch.from_numpy(lab_image)
    #lab_image = lab_image.unsqueeze(0)
    return lab_image

#function to extract the l channels and ab for training
def extract_channels(lab_image):
    # Sicherstellen, dass das LAB-Bild ein Tensor bleibt und auf der CPU verarbeitet wird
    lab_image = lab_image.cpu().clone().detach()  # Auf CPU verschieben und keine Gradienten mehr verfolgen
    
    # Extrahiere die Kanäle für die gesamte Batch
    l = lab_image[:, 0:1, :, :]  # L-Kanal, erhaltene Form: [Batchsize, 1, H, W]
    a = lab_image[:, 1:2, :, :]  # a-Kanal, erhaltene Form: [Batchsize, 1, H, W]
    b = lab_image[:, 2:2 + 1, :, :]  # b-Kanal, erhaltene Form: [Batchsize, 1, H, W]
    
    return l, a, b
    """
    l_channels = []
    a_channels = []
    b_channels = []
    #print(type(lab_image))
    lab_image = lab_image.squeeze(0)
    lab_image = lab_image.cpu().clone().detach().numpy() #converting to numpy
    #print(lab_image.shape)
    #print(lab_image[0].shape)
    #print(lab_image[2])
    l, a, b = lab_image[0], lab_image[1], lab_image[2]

    l = torch.from_numpy(l).unsqueeze(0).unsqueeze(0)
    a = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
    b = torch.from_numpy(b).unsqueeze(0).unsqueeze(0)
    #for i in lab_image:
    #print(type(lab_image))
    #lab_image = lab_image.squeeze(0)
    #lab_image = lab_image.cpu().clone().detach().numpy() #converting to numpy
    #print(lab_image.shape)
        #print(i.shape)
    #l,a,b = cv2.split(lab_image)
    #l_channels.append(l)
    #a_channels.append(a)
    #b_channels.append(b)
    return l, a, b
    #return torch.tensor(l),torch.tensor(a), torch.tensor(b)
    #return np.array(l_channels), np.array(a_channels), np.array(b_channels)
    """
# Model definition
class ConvNet(nn.Module):
    def __init__(self, num_classes = NUM_CLASSES):
        super(ConvNet, self).__init__()
        self.batchnorm8 = nn.BatchNorm2d(8)
        self.batchnorm16 = nn.BatchNorm2d(16)
        self.batchnorm64 = nn.BatchNorm2d(64)
        self.batchnorm128 = nn.BatchNorm2d(128)
        self.batchnorm256 = nn.BatchNorm2d(256)
        self.batchnorm512 = nn.BatchNorm2d(512)
        self.batchnorm1024 = nn.BatchNorm2d(1024)

        self.conv0 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.conv5_5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2)
        
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv6_5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv9 = nn.Conv2d(64, num_classes*2, kernel_size=3, padding=1) #brauchen nur a und b werte vorhersagen

    def forward(self, x):
        
        x = self.batchnorm8(F.relu(self.conv0(x)))
        x = self.batchnorm16(F.relu(self.conv1(x)))
        x = self.batchnorm64(F.relu(self.conv2(x)))
        x = self.batchnorm128(F.relu(self.conv3(x)))
        x = self.batchnorm256(F.relu(self.conv4(x)))
        x = self.batchnorm512(F.relu(self.conv5(x)))
        x = self.batchnorm1024(F.relu(self.conv5_5(x)))

        x = self.upsample0(x)
        x = self.batchnorm512(F.relu(self.conv6(x)))
        x = self.upsample1(x)
        x = self.batchnorm256(F.relu(self.conv6_5(x)))
        x = self.upsample2(x)
        x = self.batchnorm128(F.relu(self.conv7(x)))
        x = self.upsample3(x)
        x = self.batchnorm64(F.relu(self.conv8(x)))
        x = self.upsample4(x)
        x = self.batchnorm512(F.relu(self.conv9(x)))
        """
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv5_5(x))
        
        x = self.upsample0(x)
        x = F.relu(self.conv6(x))
        x = self.upsample1(x)
        x = F.relu(self.conv6_5(x))
        x = self.upsample2(x)
        x = F.relu(self.conv7(x))
        x = self.upsample3(x)
        x = F.relu(self.conv8(x))
        x = self.upsample4(x)
        x = self.conv9(x)
        """
        return x
    
runs_path = None#"Imagenette_pixel_classification_lab_20250103_015413"
#trained_model = "./models_Imagenette_classification_lab/model_20250103_015413_49"

model = ConvNet(num_classes=NUM_CLASSES).to(device)
#model.load_state_dict(torch.load(trained_model, weights_only="True"))
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

def compute_loss(outputs, targets):
    # Aufteilen der Logits in die Kanäle a und b
    a_channel_logits, b_channel_logits = torch.chunk(outputs, chunks=2, dim=1)
    
    # Sicherstellen, dass die Dimensionen korrekt sind
    assert a_channel_logits.shape[1] == NUM_CLASSES, f"Erwartet {NUM_CLASSES}, aber erhalten {a_channel_logits.shape[1]}"
    assert b_channel_logits.shape[1] == NUM_CLASSES, f"Erwartet {NUM_CLASSES}, aber erhalten {b_channel_logits.shape[1]}"
    
    # Zielwerte für die Kanäle extrahieren
    a_targets = targets[:, 0, :, :]  # Zielwerte für den a-Kanal
    b_targets = targets[:, 1, :, :]  # Zielwerte für den b-Kanal
    
    # Berechnung des Cross-Entropy-Loss für jeden Kanal
    a_channel_loss = criterion(a_channel_logits, a_targets)  # Loss für den a-Kanal
    b_channel_loss = criterion(b_channel_logits, b_targets)  # Loss für den b-Kanal
    
    # Durchschnittlicher Loss
    return (a_channel_loss + b_channel_loss) / 2

    """
    a_channel_logits, b_channel_logits = torch.chunk(outputs, chunks=2, dim=1) #teilt torch.Size([36, 768, 32, 32]) in 3X torch.Size([36, 256, 32, 32]) auf
    assert a_channel_logits.shape[1] == NUM_CLASSES
    assert b_channel_logits.shape[1] == NUM_CLASSES
    #red_logits, green_logits, blue_logits = torch.chunk(outputs, chunks=3, dim=1) #teilt torch.Size([36, 768, 32, 32]) in 3X torch.Size([36, 256, 32, 32]) auf
    #red_logits = red_logits.clamp(0,255)
    #print("\n logits")
    #print(red_logits.min())
    #print(red_logits.max())
    #print(targets.min(), targets.max())
    a_targets = targets[:, 0, :, :] # weil Rot-Wert dder pixel hier gespeichert
    b_targets = targets[:, 1, :, :] # weil Grün-Wert dder pixel hier gespeichert
    #blue_targets = targets[:, 2, :, :] # weil Blau-Wert dder pixel hier gespeicher

    #computing CrossEntropyLoss for all colors
    a_channel_loss = criterion( a_channel_logits,a_targets )
    b_channel_loss = criterion( b_channel_logits,b_targets)

    #print(f"a channel {a_channel_loss}")
    return (a_channel_loss + b_channel_loss) / 2
    """
# Training and validation functions
def train_one_epoch(epoch_index, writer):
    running_loss = 0.0
    for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch_index}"):
        #lab_image = rgb_to_lab(images)
        labels = labels.to(device)
        l_channel, a_channel, b_channel = extract_channels(images)
        l_channel, a_channel, b_channel = l_channel.to(device), a_channel.to(device), b_channel.to(device)

        optimizer.zero_grad()
        outputs = model(l_channel)
        
        #a_channel_classes = a_channel_logits.argmax(dim=1)
        #b_channel_classes = b_channel_logits.argmax(dim=1)
        #colorized_output = torch.stack([l_channel, a_channel_logits, b_channel_logits])
        #rgb_output = lab_to_rgb(colorized_output)
        loss = compute_loss(outputs, labels) 
        #loss = criterion(outputs, colorized_output)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() 
        #print(f"running")

    avg_loss = running_loss / len(train_loader) 
    writer.add_scalar('Loss/train', avg_loss, epoch_index)
    return avg_loss

def validation_loss(epoch_index, writer):
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, (vimages, vlabels) in enumerate(test_loader):
            vlabels = vlabels.to(device)
            vimages = vimages.to(device)
            l_channel, a_channel, b_channel = extract_channels(vimages)
            l_channel, a_channel, b_channel = l_channel.to(device), a_channel.to(device), b_channel.to(device)

            voutputs = model(l_channel)
            vloss =  compute_loss(voutputs, vlabels)  
            running_vloss += vloss.item() 

    avg_vloss = running_vloss / len(test_loader) 
    writer.add_scalar('Loss/validate', avg_vloss, epoch_index)
    return avg_vloss

# Training the model
def train_model():
    start_epoch = 0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if runs_path != None:
        writer = SummaryWriter('runs_lab/{}'.format(runs_path))
    else:
        writer = SummaryWriter('runs_lab/Imagenette_pixel_classification_lab_{}'.format(timestamp))

    best_vloss = float('inf')

    for epoch in tqdm(range(start_epoch,EPOCHS), desc="Progress for complete training (percentage of finished EPOCHS)"):
        print('EPOCH {}:'.format(epoch))

        model.train(True).to(device)
        avg_loss = train_one_epoch(epoch, writer)
        avg_vloss = validation_loss(epoch, writer)

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = './models_Imagenette_classification_lab/model_{}_{}'.format(timestamp, epoch)

            checkpoint = { 
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, model_path)
            #torch.save(model.state_dict(), model_path)
            print(f"saved checkpoint in {model_path}")

# Visualization
def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    return image


def plot_images(original, grayscale, colorized):
    #original_img = im_convert(original[0])
    print(original.shape)
    original_img = original[0].squeeze(0).cpu().permute(1, 2, 0).numpy()
    print(f"grayscalew {original_img.shape}")
    #original_img =  cv2.cvtColor(original_img, cv2.COLOR_LAB2RGB)
    grayscale_img = grayscale[0].cpu().numpy()[0]
    #colorized_img = im_convert(colorized)
    colorized_img = colorized[0].squeeze(0).cpu().permute(1, 2, 0).numpy()
    #colorized_img = colorized.cpu().clone().detach().numpy()
    original_img = lab2rgb(original_img)
    colorized_img = lab2rgb(colorized_img)
    print(f"grayscalew {colorized_img.shape}")
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
        for i, (images, _) in enumerate(test_loader):
            # Extrahieren der Kanäle aus dem LAB-Bild
            l_channel, a_channel, b_channel = extract_channels(images)
            l_channel, a_channel, b_channel = l_channel.to(device), a_channel.to(device), b_channel.to(device)
            
            # Normalisierung / Denormalisierung der LAB-Kanäle
            l_channel *= 100.0
            a_channel -= 128
            b_channel -= 128
            
            # Vorhersagen des Modells
            outputs = model(l_channel)
            a_channel_logits, b_channel_logits = torch.chunk(outputs, chunks=2, dim=1)
            
            # Berechnen der Klassenvorhersagen
            a_classes = a_channel_logits.argmax(dim=1).float() - 128
            b_classes = b_channel_logits.argmax(dim=1).float() - 128
            
            # Dimensionen anpassen
            if len(l_channel.shape) == 3:  # Falls Kanal-Dimension fehlt
                l_channel = l_channel.unsqueeze(1)
            if len(a_channel.shape) == 3:  # Falls Kanal-Dimension fehlt
                a_channel = a_channel.unsqueeze(1)
            if len(b_channel.shape) == 3:  # Falls Kanal-Dimension fehlt
                b_channel = b_channel.unsqueeze(1)
            if len(a_classes.shape) == 3:  # Falls Kanal-Dimension fehlt
                a_classes = a_classes.unsqueeze(1)
            if len(b_classes.shape) == 3:  # Falls Kanal-Dimension fehlt
                b_classes = b_classes.unsqueeze(1)
            
            # Stapeln der Kanäle zu LAB-Bildern
            original_lab = torch.cat([l_channel, a_channel, b_channel], dim=1)
            colorized_lab = torch.cat([l_channel, a_classes, b_classes], dim=1)
            
            print(f"a_channel: {a_channel.min(), a_channel.max(), a_channel.mean()}")
            print(f"b_channel: {b_channel.min(), b_channel.max()}")
            print(f"l_channel: {l_channel.min(), l_channel.max()}")
            print(f"a_classes: {a_classes.min(), a_classes.max()}")
            print(f"b_classes: {b_classes.min(), b_classes.max(), b_classes.mean()}")
            
            # Visualisierung der Bilder
            plot_images(original_lab, l_channel, colorized_lab)
            break

"""
def plot_examples(model=model):
    model.eval().to(device)
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            #images[..., 0] *= 100.0 
            #images[..., 1:] = images[..., 1:] * 255.0 - 128.0
            l_channel, a_channel, b_channel = extract_channels(images)
            l_channel, a_channel, b_channel = l_channel.to(device), a_channel.to(device), b_channel.to(device)
            #images = images.to(device)
            #images[:, :, 1:] += 128 
            l_channel*= 100.0 
            a_channel = a_channel-128#* 255.0 - 128.0
            b_channel = b_channel-128#* 255.0 - 128.0
            
            
            
            outputs = model(l_channel)
            a_channel_logits, b_channel_logits = torch.chunk(outputs, chunks=2, dim=1)
            #l_channel = l_channel.squeeze(0)
            l_channel, a_channel, b_channel = l_channel.squeeze(0), a_channel.squeeze(0), b_channel.squeeze(0)
            a_classes = a_channel_logits.argmax(dim=1)
            b_classes = b_channel_logits.argmax(dim=1)
            
            a_classes = (a_classes-128).float()#* 255.0 - 128.0
            b_classes = (b_classes-128).float()#* 255.0 - 128.0
            print(f"a:{a_channel.min(), a_channel.max(), a_channel.mean()}")
            print(f"b:{b_channel.min(), b_channel.max()}")
            print(f"L:{l_channel.min(), l_channel.max()}") #l wertebereich stimmt
            #a_classes -= 128
            #b_classes -= 128
            print(f"a:{a_classes.min(), a_classes.max()}")
            print(f"b:{b_classes.min(), b_classes.max(), b_classes.mean()}")
            images = torch.stack([l_channel, a_channel, b_channel ], dim=1).squeeze(0)
            colorized_lab = torch.stack([l_channel, a_classes, b_classes ], dim=1).squeeze(0)
            print(images.shape)
            print(colorized_lab.shape)
            #colorized_rgb = lab_to_rgb(colorized_lab)
            plot_images(images, l_channel, colorized_lab)
            break
"""