import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from skimage.color import lab2rgb

import cv2
from CIFAR10_classification import ConvNet, train_dataset, test_dataset, train_loader, test_loader, NUM_CLASSES, EPOCHS, BATCH_SIZE, LEARNING_RATE
#from Imagenette_classification import ConvNet, train_dataset, test_dataset, train_loader, test_loader, NUM_CLASSES, EPOCHS, BATCH_SIZE, LEARNING_RATE
#from lab_class import ConvNet, test_loader,train_loader, NUM_CLASSES, EPOCHS, BATCH_SIZE, LEARNING_RATE

#TRAINED_MODEL_PATH = "./models_CIFAR10_classification/model_20241229_001305_409" #ergebnisse nach 6 stunden Training
#TRAINED_MODEL_PATH = "./models_Imagenette_classification/model_20241230_085451_41" #ergebnisse nach 6 stunden Training
#TRAINED_MODEL_PATH = "./models_Imagenette_classification_lab/model_20250103_015413_49" 
TRAINED_MODEL_PATH = "./models_Imagenette_classification_lab/model_20250105_123600_2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#initializing model
"""
def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = np.transpose(image, (1, 2, 0))  # Von (C, H, W) zu (H, W, C)
    #image = np.clip(image, 0, 1)  # Normalisieren auf den Bereich [0, 1]
    return image

def plot_images(original, grayscale, colorized):
    original_img = im_convert(original[0])
    print(grayscale.shape)
    grayscale_img = grayscale[0].cpu().numpy()[0]
    
    print(grayscale_img.shape)
    colorized_img = im_convert(colorized[0])
    print(colorized_img.shape)
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


def plot_examples(model= ConvNet(num_classes=NUM_CLASSES)):
        
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, weights_only="False"))
    #optimizer = optim.Adam(model.parameters(), lr = 0.0005)
    model.eval().to(device)


    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            grayscale_images = transforms.functional.rgb_to_grayscale(images).to(device)
            images = images.to(device)
            outputs = model(grayscale_images)
            red_logits, green_logits, blue_logits = torch.chunk(outputs, chunks=3, dim=1)
            red_classes = red_logits.argmax(dim=1)
            green_classes = green_logits.argmax(dim=1)
            blue_classes = blue_logits.argmax(dim=1)
            print(f" red {red_classes.shape}")
            colorized_output = torch.stack([red_classes, green_classes, blue_classes], dim=1)
            print(images.min(), images.max())
            print(images.shape)
            print(outputs.min(), outputs.max())
            print(outputs.shape)
            images = images.clamp(0,1) #damit imshow nicht cliped später
            plot_images(images, grayscale_images, colorized_output)
            break

"""


"""
#function to extract the l channels and ab for training
def extract_channels(lab_image):
    l_channels = []
    a_channels = []
    b_channels = []
    #print(type(lab_image))
    lab_image = lab_image.squeeze(0)
    lab_image = lab_image.cpu().clone().detach().numpy() #converting to numpy
    
    l, a, b = lab_image[0], lab_image[1], lab_image[2]

    l = torch.from_numpy(l).unsqueeze(0).unsqueeze(0)
    a = torch.from_numpy(a).unsqueeze(0).unsqueeze(0)
    b = torch.from_numpy(b).unsqueeze(0).unsqueeze(0)
    return l, a, b
    #return torch.tensor(l),torch.tensor(a), torch.tensor(b)
    #return np.array(l_channels), np.a
    """
def extract_channels(lab_image):
    # Sicherstellen, dass das LAB-Bild ein Tensor bleibt und auf der CPU verarbeitet wird
    lab_image = lab_image.cpu().clone().detach()  # Auf CPU verschieben und keine Gradienten mehr verfolgen
    
    # Extrahiere die Kanäle für die gesamte Batch
    l = lab_image[:, 0:1, :, :]  # L-Kanal, erhaltene Form: [Batchsize, 1, H, W]
    a = lab_image[:, 1:2, :, :]  # a-Kanal, erhaltene Form: [Batchsize, 1, H, W]
    b = lab_image[:, 2:2 + 1, :, :]  # b-Kanal, erhaltene Form: [Batchsize, 1, H, W]
    
    return l, a, b
# Visualization
def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    return image
"""
def plot_images(original, grayscale, colorized):
    #original_img = im_convert(original[0])
    original_img = original.squeeze(0).cpu().permute(1, 2, 0).numpy()
    print(f"grayscalew {original_img.shape}")
    #original_img =  cv2.cvtColor(original_img, cv2.COLOR_LAB2RGB)
    grayscale_img = grayscale[0].cpu().numpy()
    #colorized_img = im_convert(colorized)
    colorized_img = colorized.squeeze(0).cpu().permute(1, 2, 0).numpy()
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

def plot_examples(model=ConvNet(NUM_CLASSES)):
    checkpoint = torch.load(TRAINED_MODEL_PATH)
    model.load_state_dict(checkpoint['model'])
    #model.load_state_dict(torch.load(TRAINED_MODEL_PATH, weights_only="True"))
    #optimizer = optim.AdamW(model.parameters(), lr = 0.0005)
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
            break"""
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

def plot_examples(model=ConvNet(NUM_CLASSES)):
    model.eval().to(device)
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
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