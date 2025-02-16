import cv2 as cv
import os
import torch
import torch.utils
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# from dataset_CIFAR10 import pixelclass_dataset_CIFAR10

torch.cuda.is_available()
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 80
BATCH_SIZE = 10
LEARNING_RATE = 0.00005


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0, 0, 0], [1, 1, 1])]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="./CIFAR_data", train=True, download=False, transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root="./CIFAR_data", train=False, download=False, transform=transform
)

# train_dataset = pixelclass_dataset_CIFAR10(root='./data', train=True, download=False, transform=transform)
# test_dataset = pixelclass_dataset_CIFAR10(root='./data', train=False, download=False, transform=transform)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True
)



class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2)  
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)  
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) 
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  
        
        # Bottleneck
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1) 
        
        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')  
        self.conv10 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear') 
        self.conv11 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1) 
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear') 
        self.conv13 = nn.Conv2d(32, 3, kernel_size=3, padding=1)  
    
    
    
    def forward(self,x):
        # Encoder
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv6(x))
        
        # Decoder
        x = self.upsample1(x)
        x = F.relu(self.conv10(x))
        x = self.upsample2(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.upsample3(x)
        x = F.sigmoid(self.conv13(x))
        #print(x.shape)
        
        return x
        """
        #Das hier eigentliches Netzwerk ,das obennist aber bereits trainiert
        super(ConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)

        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv7 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):

        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Decoder
        x = self.upsample1(x)
        x = F.relu(self.conv4(x))
        x = self.upsample2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.upsample3(x)
        x = F.sigmoid(self.conv7(x))

        return x
        """

# moving the model to the used device
model = ConvNet().to(device)

#criterion = nn.MSELoss().to(device)
criterion = nn.HuberLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0

    total_batches = len(train_loader)
    # print(f"Total batches in train_loader: {total_batches}")

    for i, (images, labels) in tqdm(
        enumerate(train_loader), total=total_batches, desc="Progress in current epoch"
    ):
        grayscale_images = transforms.functional.rgb_to_grayscale(images)
        images = images.to(device)
        inputs = grayscale_images.to(device)

        # Debug input shape
        # print(f"Input shape to model: {inputs.shape}")

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions
        outputs = model(inputs)

        loss = criterion(outputs, images)

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN encountered in loss at batch {i}")
            break

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f"Batch {i + 1}, Loss: {last_loss}")
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return running_loss * 10000.0


def trainConvNet():
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs_CIFAR_regression/CIFAR_colorization_HuberLoss_{}".format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.0

    # model = ConvNet().to(device)

    # for epoch in range(EPOCHS):
    for epoch in tqdm(
        range(EPOCHS),
        desc="Progress for complete training (percentage of finished EPOCHS)",
    ):
        print("EPOCH {}:".format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, (vimages, _) in enumerate(test_loader):
                if i >= len(test_loader):
                    break

                vimages = vimages.to(device)
                vimages_grayscale = transforms.functional.rgb_to_grayscale(vimages)
                vinputs = vimages_grayscale.to(device)

                voutputs = model(vinputs).to(device)
                vloss = criterion(voutputs, vimages)

                # Debug validation loss
                if torch.isnan(vloss):
                    print(f"NaN encountered in validation loss at batch {i}")
                    break

                running_vloss += vloss.item()

        avg_vloss = running_vloss / (i + 1)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

            # saving model each epoch
            model_path = "./models_Cifar10/model_{}_{}".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
            print(f"saved checkpoint in {model_path}")

        epoch_number += 1


# Funktion zum Umwandeln eines Torch-Tensors in ein Numpy-Array
def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = np.transpose(image, (1, 2, 0))  # Von (C, H, W) zu (H, W, C)
    return image


# Visualisierungsfunktion
def plot_images(original, grayscale, colorized):
    original_img = im_convert(original[0])  # Originalbild (RGB)
    grayscale_img = im_convert(grayscale[0])  # Graustufenbild
    colorized_img = im_convert(colorized[0])  # Vom Modell eingefärbtes Bild

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Originalbild
    ax[0].imshow(original_img)
    ax[0].set_title("Original RGB Image")
    ax[0].axis("off")

    # Grauwertbild
    ax[1].imshow(grayscale_img, cmap="gray")
    ax[1].set_title("Grayscale Image")
    ax[1].axis("off")

    # Vom Modell eingefärbtes Bild
    ax[2].imshow(colorized_img)
    ax[2].set_title("Colorized by Model")
    ax[2].axis("off")

    print(original.min, original.max)
    plt.show()

# Beispiel: Zeige ein Bild, graues Bild und das durch das Modell eingefärbte Bild an
def plot_examples(model=model):
    try:
        model.eval()  # Schalte das Modell in den Evaluierungsmodus
    except (NameError, AttributeError) as e:
        model.eval().to(device)

    # if images != None:
    with torch.no_grad():  # Keine Gradient-Berechnung nötig
        for i, (images, _) in enumerate(train_loader):
            grayscale_images = transforms.functional.rgb_to_grayscale(images).to(device)
            images = images.to(device)
            # Vorhersage durch das Modell
            outputs = model(grayscale_images)

            # Werte auf 0 bis 255 umrechnen
            outputs = (outputs * 255).to(torch.uint8)
            # Plotte das Original, das Grauwertbild und das eingefärbte Bild
            plot_images(images, grayscale_images, outputs)

            break  
                