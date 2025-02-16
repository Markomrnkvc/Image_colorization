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
from skimage.color import lab2rgb, rgb2lab
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def colorization(args):


    def im_convert(tensor):
        image = tensor.cpu().clone().detach().numpy()
        image = np.transpose(image, (1, 2, 0))  # Von (C, H, W) zu (H, W, C)
        #image = np.clip(image, 0, 1)  # Normalisieren auf den Bereich [0, 1]
        return image

    def plot_images(original, grayscale, colorized):
        original_img = im_convert(original[0])
        grayscale_img = grayscale[0].cpu().numpy()[0]
        
        colorized_img = im_convert(colorized[0])
        fig, ax = plt.subplots(1, 4, figsize=(24, 8))
        lab_colorized = rgb2lab(colorized_img/255)
        lab_original = rgb2lab(original_img)

        l_original, a_original, b_original = extract_channels(lab_original)
        l_colorized, a_colorized, b_colorized = extract_channels(lab_colorized)
        print(original_img.shape)
        print(colorized_img.shape)
        colorized_lab = np.stack([l_original, a_colorized, b_colorized], axis=-1)
        colorized_lab = lab2rgb(colorized_lab)
        print(colorized_lab.mean())
        print(colorized_lab.shape)
        ax[0].imshow(original_img)
        ax[0].set_title("Original RGB Image")
        ax[0].axis('off')
        ax[1].imshow(grayscale_img, cmap='gray')
        ax[1].set_title("Grayscale Image")
        ax[1].axis('off')
        ax[2].imshow(colorized_img)
        ax[2].set_title("Colorized by Model")
        ax[3].axis('off')
        ax[3].imshow(colorized_lab)
        ax[3].set_title("Colorized by Model LAB")
        ax[3].axis('off')
        plt.show()


    def plot_examples(dataset,model):
        
        if dataset == "Cifar10":
            model.load_state_dict(torch.load(TRAINED_MODEL_PATH, weights_only="False"))
            optimizer = optim.Adam(model.parameters(), lr = 0.0005)
            model.eval().to(device)
        elif args.dataset == "Imagenette":
            checkpoint = torch.load(TRAINED_MODEL_PATH)
            model.load_state_dict(checkpoint['model'])
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
                
                images = images.clamp(0,1) #damit imshow nicht cliped später
                plot_images(images, grayscale_images, colorized_output)
                break

    def extract_channels(lab_image):
        # Extrahiere die Kanäle für die gesamte Batch
        l = lab_image[:, :, 0]  # L-Kanal, erhaltene Form: [Batchsize, 1, H, W]
        a = lab_image[:, :, 1]  # a-Kanal, erhaltene Form: [Batchsize, 1, H, W]
        b = lab_image[:, :, 2]  # b-Kanal, erhaltene Form: [Batchsize, 1, H, W]
        return l, a, b
    

    
    if args.dataset == "Cifar10":
        from CIFAR10_classification import ConvNet, train_dataset, test_dataset, train_loader, test_loader, NUM_CLASSES, EPOCHS, BATCH_SIZE, LEARNING_RATE
        TRAINED_MODEL_PATH = "./models_CIFAR10_classification/model_20241229_001305_409"
        print("Cifar")

        plot_examples(dataset = "Cifar10", model = ConvNet())

    elif args.dataset == "Imagenette":
        from Imagenette_classification import ConvNet, train_dataset, test_dataset, train_loader, test_loader, NUM_CLASSES, EPOCHS, BATCH_SIZE, LEARNING_RATE
        TRAINED_MODEL_PATH = "./models_Imagenette_classification/model_20250109_072447_23"
        print("imagenette")
        plot_examples(dataset = "Imagenette", model = ConvNet())

