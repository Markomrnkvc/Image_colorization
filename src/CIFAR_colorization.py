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

torch.cuda.is_available()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 100
BATCH_SIZE = 1000
LEARNING_RATE = 0.005


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0,0,0], [1,1,1])
                                ])

train_dataset = torchvision.datasets.CIFAR10(root = "./CIFAR_data", train=True, download=False, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root = "./CIFAR_data", train=False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

"""
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
"""
    

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        """
        self.conv1_16 = nn.Conv2d(1, 16,3, padding=1)
        self.conv32_16 = nn.Conv2d(32, 16,3, padding=1)
        self.conv16_8= nn.Conv2d(16, 8,3, padding=1)
        self.conv8_16 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1)
        self.conv16_16 = nn.Conv2d(16, 16,3, padding=1)
        self.conv16_32 = nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1)
        self.conv32_32_2 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.conv32_64 = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1)
        self.conv64_128 = nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1)
        self.conv128_64 = nn.Conv2d(128, 64,3, padding=1)
        self.conv64_32 = nn.Conv2d(64,32,3, padding=1)
        self.conv8_3 = nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1)
        """
        """
        self.conv1_16 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv16_32 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv32_16 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv16_8 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        
        # Transpose Convolutional layers for upsampling
        self.upconv8_16 = nn.ConvTranspose2d(8, 16, kernel_size=2, stride=2)  # Upsample by factor of 2
        self.upconv16_32 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2) # Upsample by factor of 2
        self.upconv32_3 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)   # Final upsample for output
        """
        
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # Input: [1, 32, 32], Output: [64, 32, 32]
        self.conv2 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2)  # Output: [64, 16, 16]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: [128, 16, 16]
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)  # Output: [128, 8, 8]
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: [256, 8, 8]
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)  # Output: [256, 4, 4]
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Output: [512, 4, 4]
        
        # Bottleneck
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # Output: [256, 4, 4]
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # Output: [128, 4, 4]
        
        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')  # Output: [128, 8, 8]
        self.conv10 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # Output: [64, 8, 8]
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')  # Output: [64, 16, 16]
        self.conv11 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # Output: [32, 16, 16]
        self.conv12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)  # Output: [3, 16, 16]
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')  # Output: [3, 32, 32]
        self.conv13 = nn.Conv2d(32, 3, kernel_size=3, padding=1)  # Output: [3, 16, 16]
       
        
    
    def forward(self,x):
        """
        print(x.shape)
        x = F.relu(self.conv1_16(x))
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv16_32(x))
        print(x.shape)
        x = F.relu(self.conv32_16(x))
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv16_8(x))
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv8_16(x))
        print(x.shape)
        x = F.relu(self.conv16_32(x))
        #x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv32_64(x))
        print(x.shape)
        x = F.relu(self.conv64_128(x))
        print(x.shape)
        x = F.relu(self.conv128_64(x))
        print(x.shape)
        x = F.relu(self.conv64_32(x))
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv32_16(x))
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv16_8(x))
        x = self.pool(x)
        print(f"output size: {x.shape}")
        x = torch.sigmoid(self.conv8_3(x))
        print(f"output size: {x.shape}")
        """
        #print(f"Input size: {x.shape}")
        """
        # Downsampling
        x = F.relu(self.conv1_16(x))
        x = self.pool(x)
        #print(f"After conv1_16 + pool: {x.shape}")
        
        x = F.relu(self.conv16_32(x))
        #print(f"After conv16_32: {x.shape}")
        
        x = F.relu(self.conv32_16(x))
        x = self.pool(x)
        #print(f"After conv32_16 + pool: {x.shape}")
        
        x = F.relu(self.conv16_8(x))
        x = self.pool(x)
        #print(f"After conv16_8 + pool: {x.shape}")
        
        # Upsampling
        x = F.relu(self.upconv8_16(x))  # Transpose Convolution
        #print(f"After upconv8_16: {x.shape}")
        
        x = F.relu(self.upconv16_32(x))  # Transpose Convolution
        #print(f"After upconv16_32: {x.shape}")
        
        x = torch.sigmoid(self.upconv32_3(x))  # Final output
        #print(f"Output size: {x.shape}")
        """
        
        #nach vorlage asu github
        # Encoder
        #x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        #x = F.relu(self.conv7(x))
        
        # Bottleneck
        #x = F.relu(self.conv8(x))
        #x = F.relu(self.conv9(x))
        
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
#moving the model to the used device
model = ConvNet().to(device) 

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)


"""
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (images,_) in tqdm(enumerate(train_loader), total=len(train_dataset), desc="Progress in current epoch (#batch/total batches)"):
    #for i, (images,_) in enumerate(train_loader):
        #grayscaling images for training
        grayscale_images = transforms.functional.rgb_to_grayscale(images)
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

        if i % 10000 == 9999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss"""
        

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    total_batches = len(train_loader)
    #print(f"Total batches in train_loader: {total_batches}")

    for i, (images, _) in tqdm(enumerate(train_loader), total=total_batches, desc="Progress in current epoch"):
        grayscale_images = transforms.functional.rgb_to_grayscale(images)
        images = images.to(device)
        inputs = grayscale_images.to(device)

        # Debug input shape
        #print(f"Input shape to model: {inputs.shape}")

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
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return running_loss * 10000.0

def trainConvNet():
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/CIFAR_colorization_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.

    #model = ConvNet().to(device)

    #for epoch in range(EPOCHS):
    for epoch in tqdm(range(EPOCHS),desc="Progress for complete training (percentage of finished EPOCHS)"):
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
            """
            for i,  (vimages,_) in enumerate(test_loader):
                vimages = vimages.to(device)
                vimages_grayscale = transforms.functional.rgb_to_grayscale(vimages)
                vinputs = vimages_grayscale.to(device)
                voutputs = model(vinputs).to(device)
                vloss = criterion(voutputs, vimages)
                running_vloss += vloss
            """
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
            grayscale_images = transforms.functional.rgb_to_grayscale(images).to(device)
            images = images.to(device)
            # Vorhersage durch das Modell
            outputs = model(grayscale_images)
            
            #Werte auf 0 bis 255 umrechnen
            outputs = (outputs * 255).to(torch.uint8)
            # Plotte das Original, das Grauwertbild und das eingefärbte Bild
            plot_images(images, grayscale_images, outputs)

            break  # Nur für ein Beispiel, damit die Schleife nicht endlos läuft"""

