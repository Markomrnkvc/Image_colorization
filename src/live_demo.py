
import cv2 as cv
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from skimage.color import lab2rgb, rgb2lab
import torchvision.transforms as transforms

NUM_CLASSES = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


def extract_channels(lab_image):
    # Extrahiere die Kanäle für die gesamte Batch
    l = lab_image[:, :, 0]  # L-Kanal, erhaltene Form: [Batchsize, 1, H, W]
    a = lab_image[:, :, 1]  # a-Kanal, erhaltene Form: [Batchsize, 1, H, W]
    b = lab_image[:, :, 2]  # b-Kanal, erhaltene Form: [Batchsize, 1, H, W]
    return l, a, b

def live(args):
    if args.dataset == "Cifar10":
        from CIFAR10_classification import ConvNet, NUM_CLASSES,  im_convert
        model = ConvNet(num_classes=NUM_CLASSES).to(device)
        trained_model_path = "./models_CIFAR10_classification/model_20241229_001305_409"
        
        model.load_state_dict(torch.load(trained_model_path, weights_only="False"))
        #optimizer = optim.Adam(model.parameters(), lr = 0.0005)
        model.eval().to(device)
        
    elif args.dataset == "Imagenette":
        #from Imagenette_classification import ConvNet
        from Imagenette_classification import ConvNet,  NUM_CLASSES, im_convert

        trained_model_path = "./models_Imagenette_classification/model_20250109_072447_23"
        # Load the model checkpoint from a previous training session (check code in train.py)
        model = ConvNet(num_classes=NUM_CLASSES).to(device)
        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint['model'])
        model.eval().to(device)
    

    # Create a video capture device to retrieve live footage from the webcam.
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Webcam did not open")
        

    # Calculate the border sizes once
    ret, frame = cap.read()
    if not ret:
        raise IOError("No Image found")
    
    # for continuous processing of the frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            raise IOError("No Image found")
        
        #resizing image to 36X36 if Cifar10-model is being used
        if args.dataset == "Cifar10":
            frame = cv.resize(frame, (36, 36), interpolation=cv.INTER_LANCZOS4)
        #frame = cv.resize(frame, (256, 256))
        frame = Image.fromarray(frame)
        frame_tensor = transform(frame)

        grayscale_images = transforms.functional.rgb_to_grayscale(frame_tensor).to(device)

        #adding batch
        grayscale_images = grayscale_images.unsqueeze(0) #mein Model erwartet input in form [b,c,w,h]
        
        #colorizing with model
        outputs = model(grayscale_images.to(device))
        red_logits, green_logits, blue_logits = torch.chunk(outputs, chunks=3, dim=1)
        red_classes = red_logits.argmax(dim=1)
        green_classes = green_logits.argmax(dim=1)
        blue_classes = blue_logits.argmax(dim=1)
        colorized_output = torch.stack([red_classes, green_classes, blue_classes], dim=1)


        grayscale_images = grayscale_images[0].cpu().numpy()[0]
        colorized_image = im_convert(colorized_output[0])#erstes Bild des Batches
        original_img = im_convert(frame_tensor)
        #colorized_image = colorized_image.astype(np.uint8)
        
        #transfering to LAB-SPace
        lab_colorized = rgb2lab(colorized_image/255)
        lab_original = rgb2lab(original_img)

        #extracting channels from LAB-Images
        l_original, a_original, b_original = extract_channels(lab_original)
        l_colorized, a_colorized, b_colorized = extract_channels(lab_colorized)

        #combining gray input with colorized channels
        colorized_lab = np.stack([l_original, a_colorized, b_colorized], axis=-1)
        colorized_lab = lab2rgb(colorized_lab)
        
        #resizing image windows
        cv.namedWindow('Input', cv.WINDOW_NORMAL)
        cv.resizeWindow('Input', 600, 500)
        cv.namedWindow('grayscale', cv.WINDOW_NORMAL)
        cv.resizeWindow('grayscale', 600, 500)
        cv.namedWindow('colorized', cv.WINDOW_NORMAL)
        cv.resizeWindow('colorized', 600, 500)

        cv.imshow('Input',original_img)
        cv.imshow('grayscale', grayscale_images)
        cv.imshow('colorized', colorized_lab)

        c = cv.waitKey(1)
        if c == 10:
            break
       

    # Release the video capture resources and close all OpenCV windows.
    cap.release()
    cv.destroyAllWindows()
    

""""""
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from skimage.color import rgb2lab, lab2rgb,rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
from Imagenette_classification import ConvNet,  NUM_CLASSES, im_convert

trained_model_path = "./models_Imagenette_classification/model_20250109_072447_23"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Load the model checkpoint from a previous training session (check code in train.py)
model = ConvNet(num_classes=256).to(device)
checkpoint = torch.load(trained_model_path)
model.load_state_dict(checkpoint['model'])
model.eval().to(device)

# Transformationen für die Bilder
def load_and_transform_images(root, transform=None):
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    transformed_images = []
    #original_images = []

    for file_name in os.listdir(root):
        file_path = os.path.join(root, file_name)
        if file_name.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
            # Bild laden
            image = Image.open(file_path).convert("RGB")
            #original_images.append(np.array(image))

            # Transformation anwenden
            transformed_images.append(transform(image))

    return transformed_images#, original_images


# Farben aus LAB extrahieren
def extract_channels(lab_image):
    l = lab_image[..., 0]  # L-Kanal
    a = lab_image[..., 1]  # a-Kanal
    b = lab_image[..., 2]  # b-Kanal
    return l, a, b


# Visualisierung der Ergebnisse
def visualize_results(original_images, grayscale_images, colorized_images):
    for original, gray, colorized in zip(original_images, grayscale_images, colorized_images):
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        # Originalbild
        ax[0].imshow(original)
        ax[0].set_title("Original Image")
        ax[0].axis('off')

        # Graustufenbild
        ax[1].imshow(gray, cmap='gray')
        ax[1].set_title("Grayscale Image")
        ax[1].axis('off')

        # Colorized Image
        ax[2].imshow(colorized)
        ax[2].set_title("Colorized Image")
        ax[2].axis('off')

        plt.show()

def visualize_results_dynamic(original, gray, colorized):
    plt.figure(figsize=(18, 6))

    # Originalbild
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')

    # Graustufenbild
    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')

    # Colorized Image
    plt.subplot(1, 3, 3)
    plt.imshow(colorized)
    plt.title("Colorized Image")
    plt.axis('off')

    plt.pause()  # Pause für 2 Sekunden
    plt.clf()  # Clear the figure für das nächste Bild

from matplotlib.animation import FuncAnimation
from skimage.color import rgb2lab, lab2rgb
import numpy as np
import torch

# Funktionen und Variablen
frames_data = []  # Zum Speichern der Bilddaten

# Dynamische Visualisierung mit matplotlib.animation
def update_plot(frame_index):
    #Aktualisiert den Plot mit neuen Bilddaten.
    original, gray, colorized = frames_data[frame_index]

    # Originalbild
    ax[0].imshow(original)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    # Graustufenbild
    ax[1].imshow(gray, cmap='gray')
    ax[1].set_title("Grayscale Image")
    ax[1].axis('off')

    # Colorized Image
    ax[2].imshow(colorized)
    ax[2].set_title("Colorized Image")
    ax[2].axis('off')


# Hauptfunktion
def main():
    global ax, frames_data

    # Pfad zum Ordner mit Bildern
    image_folder = "C:/Users/marko/Pictures/Camera Roll/"

    # Transformationen definieren
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Bilder laden und transformieren
    transformed_images = load_and_transform_images(image_folder, transform)

    # Bildverarbeitung vorbereiten
    for image_tensor in transformed_images:
        # Graustufenbild extrahieren
        gray_image = transforms.functional.rgb_to_grayscale(image_tensor)
        gray_image = gray_image.float().unsqueeze(0)  # [b, c, w, h]

        # Colorizing with  model
        outputs = model(gray_image.to(device))
        red_logits, green_logits, blue_logits = torch.chunk(outputs, chunks=3, dim=1)
        red_classes = red_logits.argmax(dim=1)
        green_classes = green_logits.argmax(dim=1)
        blue_classes = blue_logits.argmax(dim=1)
        colorized_output = torch.stack([red_classes, green_classes, blue_classes], dim=1)

        # Konvertiere Tensoren zu Bildern
        gray_image_np = gray_image[0].cpu().numpy()[0]
        image_tensor_np = im_convert(image_tensor.clamp(0, 1))
        colorized_image_np = im_convert(colorized_output[0])

        # Transferiere zu LAB und kombiniere
        lab_original = rgb2lab(image_tensor_np)
        lab_colorized = rgb2lab(colorized_image_np / 255)
        l_original, a_original, b_original = extract_channels(lab_original)
        l_colorized, a_colorized, b_colorized = extract_channels(lab_colorized)
        colorized_lab = np.stack([l_original, a_colorized, b_colorized], axis=-1)
        colorized_lab = lab2rgb(colorized_lab)

        # Speichere die Bilddaten für die Animation
        frames_data.append((image_tensor_np, gray_image_np, colorized_lab))

    # Erstelle die Animation
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    anim = FuncAnimation(
        fig,
        update_plot,
        frames=len(frames_data),
        interval=50, 
        repeat=True
    )
    plt.show()
