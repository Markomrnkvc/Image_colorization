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


torch.cuda.is_available()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 4
BATCH_SIZE = 150
LEARNING_RATE = 0.005


transform = transforms.Compose([transforms.ToTensor()
                                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])

train_dataset = torchvision.datasets.CIFAR10(root = "./CIFAR_data", train=True, download=False, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root = "./CIFAR_data", train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.ConvTranspose2d(32, 64, 3, stride = 2)
        self.conv3 = nn.ConvTranspose2d(64, 128, 4, stride = 4)
        self.conv4 = nn.ConvTranspose2d(128, 64, 3)
        #self.conv6 = nn.ConvTranspose2d(256, 256, 3, stride = 2)
        self.conv5 = nn.Conv2d(64, 3, 1)
        
        #self.batnchnorm32 = nn.BatchNorm2d(32)
        #self.batnchnorm64 = nn.BatchNorm2d(64)
        #self.batnchnorm128 = nn.BatchNorm2d(128)
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #x = self.batnchnorm32(F.relu(self.conv1(x)))
        #x = F.relu(self.batnchnorm32(self.conv1(x)))
        #print(x.shape)
        x = self.pool(x)     
        #print(f"nach pooling {x.shape}")  
        x = F.relu(self.conv2(x))
        #x = self.batnchnorm64(F.relu(self.conv2(x)))
        #x = F.relu(self.batnchnorm64(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        #x = nn.BatchNorm2d(64) #batchnormalization
        #print(x.shape)  
        #x = self.pool(x)            
        #print(f"nach pooling {x.shape}")
        x = F.relu(self.conv3(x))
        #x = self.batnchnorm128(F.relu(self.conv3(x)))
        #x = F.relu(self.batnchnorm128(self.conv3(x)))
        #x = F.relu(self.conv3(x))
        #print(x.shape)  
        x = self.pool(x)            
        #x = nn.BatchNorm2d(128) #batchnormalization
        #print(f"nach pooling {x.shape}")
        x = F.relu(self.conv4(x))
        #x = self.batnchnorm64(F.relu(self.conv4(x)))
        #x = F.relu(self.conv6(x)) 
        #print(x.shape)  
        x = self.pool(x)         
        #x = nn.BatchNorm2d(64) #batchnormalization 
        #print(x.shape)  
        #x = F.relu(self.conv5(x))  
        x = torch.sigmoid(self.conv5(x))
        #print(f"nach sigmoid {x.shape}")
        return x

#moving the model to the used device
model = ConvNet().to(device) 

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

def rgb_to_gray(img):
    return img.mean(dim=1, keepdim=True)

def denormalize(tensor):
    # Annahme: tensor hat Form (C, H, W) und ist im Bereich [-1, 1] nach Normalisierung
    denorm = tensor.clone()  # Erstellen einer Kopie des Tensors, um das Original zu bewahren
    for t in range(denorm.size(0)):  # Für jeden Farbkanal
        denorm[t] = denorm[t] * 0.5 + 0.5  # Rücknormalisierung mit Standardabweichung und Mittelwert
    return denorm


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (images,_) in enumerate(train_loader):
        #grayscaling images for training
        grayscale_images = rgb_to_gray(images)
        images = images.to(device)
        # defining model inputs
        inputs= grayscale_images.to(device)


        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)
        
        # Compute the loss and its gradients
        loss = criterion(outputs, images)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
        
def trainConvNet():
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/CIFAR_colorization_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.

    #model = ConvNet().to(device)

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i,  (vimages,_) in enumerate(test_loader):
                vimages = vimages.to(device)
                vimages_grayscale = rgb_to_gray(vimages)
                vinputs = vimages_grayscale.to(device)
                voutputs = model(vinputs).to(device)
                vloss = criterion(voutputs, vimages)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            
            #saving model each epoch
            #checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
            #os.mkdir('./runs/CIFAR_colorization_{}./models'.format(timestamp))
            #model_path = './runs/CIFAR_colorization_{}./models/model_{}_{}'.format(timestamp, timestamp, epoch_number)
            model_path = './models_Cifar10/model_{}_{}'.format( timestamp, epoch_number)
            #torch.save(model.state_dict(), model_path)
            torch.save(model.state_dict(), model_path)
            #torch.save(model, model_path)
            #torch.save(checkpoint, model_path)
            print(f"saved checkpoint in {model_path}")

        epoch_number += 1
    




# Funktion zum Umwandeln eines Torch-Tensors in ein Numpy-Array
def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = np.transpose(image, (1, 2, 0))  # Von (C, H, W) zu (H, W, C)
    #image = np.clip(image, 0, 1)  # Werte zwischen 0 und 1 setzen
    return image

# Visualisierungsfunktion
def plot_images(original, grayscale, colorized):
    original_img = im_convert(original[0])  # Originalbild (RGB)
    grayscale_img = im_convert(grayscale[0])  # Graustufenbild
    colorized_img = im_convert(colorized[0])  # Vom Modell eingefärbtes Bild
    print(colorized_img.mean())
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Originalbild
    ax[0].imshow(original_img)
    ax[0].set_title("Original RGB Image")
    ax[0].axis('off')
    
    # Grauwertbild
    ax[1].imshow(grayscale_img, cmap='gray')
    ax[1].set_title("Grayscale Image")
    ax[1].axis('off')
    
    # Vom Modell eingefärbtes Bild
    ax[2].imshow(colorized_img)
    ax[2].set_title("Colorized by Model")
    ax[2].axis('off')
    
    
    print(original.min, original.max)
    plt.show()

def denormalize(tensor):
    # Annahme: tensor hat Form (C, H, W) und ist im Bereich [-1, 1] nach Normalisierung
    denorm = tensor.clone()  # Erstellen einer Kopie des Tensors, um das Original zu bewahren
    for t in range(denorm.size(0)):  # Für jeden Farbkanal
        denorm[t] = denorm[t] * 0.5 + 1  # Rücknormalisierung mit Standardabweichung und Mittelwert
    return denorm

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                    std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                            transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                    std = [ 1., 1., 1. ]),
                            ])

# Beispiel: Zeige ein Bild, graues Bild und das durch das Modell eingefärbte Bild an
def plot_examples(model = model):
    try:
        model.eval()  # Schalte das Modell in den Evaluierungsmodus
    except (NameError,AttributeError) as e:
        model.eval().to(device)

    #if images != None:
    with torch.no_grad():  # Keine Gradient-Berechnung nötig
        for i, (images, _) in enumerate(train_loader):
            #images = denormalize(images)
            grayscale_images = rgb_to_gray(images).to(device)
            images = images.to(device)
            # Vorhersage durch das Modell
            outputs = model(grayscale_images)
            #images = denormalize(images)
            #images = invTrans(images)
            #outputs = denormalize(outputs)
            # Plotte das Original, das Grauwertbild und das eingefärbte Bild
            plot_images(images, grayscale_images, outputs)

            break  # Nur für ein Beispiel, damit die Schleife nicht endlos läuft"""

