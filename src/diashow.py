
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
def main(args):
    global ax, frames_data

    # Pfad zum Ordner mit Bildern
    image_folder = args.folder

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
