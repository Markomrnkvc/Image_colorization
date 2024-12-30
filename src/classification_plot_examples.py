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
#from CIFAR10_classification import ConvNet, train_dataset, test_dataset, train_loader, test_loader, NUM_CLASSES, EPOCHS, BATCH_SIZE, LEARNING_RATE
from Imagenette_classification import ConvNet, train_dataset, test_dataset, train_loader, test_loader, NUM_CLASSES, EPOCHS, BATCH_SIZE, LEARNING_RATE


#TRAINED_MODEL_PATH = "./models_CIFAR10_classification/model_20241229_001305_409" #ergebnisse nach 6 stunden Training
TRAINED_MODEL_PATH = "./models_Imagenette_classification/model_20241230_085451_35" #ergebnisse nach 6 stunden Training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#initializing model


def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = np.transpose(image, (1, 2, 0))  # Von (C, H, W) zu (H, W, C)
    #image = np.clip(image, 0, 1)  # Normalisieren auf den Bereich [0, 1]
    return image

def plot_images(original, grayscale, colorized):
    original_img = im_convert(original[0])
    grayscale_img = grayscale[0].cpu().numpy()[0]
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
    optimizer = optim.Adam(model.parameters(), lr = 0.0005)
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
            colorized_output = torch.stack([red_classes, green_classes, blue_classes], dim=1)
            print(images.min(), images.max())
            print(images.shape)
            print(outputs.min(), outputs.max())
            print(outputs.shape)
            images = images.clamp(0,1) #damit imshow nicht cliped später
            plot_images(images, grayscale_images, colorized_output)
            break

