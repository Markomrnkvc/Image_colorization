import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from CIFAR_colorization import ConvNet,  test_dataset, test_loader, train_loader, train_dataset
#trained_model_path = "./models_Cifar10/model_20241104_232646_18"
#trained_model_path = "./models_Cifar10/model_20241208_193034_14"
#trained_model_path = "./models_Cifar10/model_20241208_181658_49" #das hier ist die Unet structure
trained_model_path = "./models_Cifar10/model_20241211_124608_99"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    ax[0].axis('off')

    # Grauwertbild
    ax[1].imshow(grayscale_img, cmap='gray')
    ax[1].set_title("Grayscale Image")
    ax[1].axis('off')

    # Vom Modell eingefärbtes Bild
    ax[2].imshow(colorized_img)
    ax[2].set_title("Colorized by Model")
    ax[2].axis('off')

    plt.show()


def eval_model_and_plot():
    # Beispiel: Zeige ein Bild, graues Bild und das durch das Modell eingefärbte Bild an
    try:
        model.eval()  # Schalte das Modell in den Evaluierungsmodus
    except (NameError,AttributeError) as e:
        #loading model if not loaded already
        #need to adjust the path manually
        model = ConvNet()
        model.load_state_dict(torch.load(trained_model_path, weights_only="True"))
        model.eval().to(device)

    with torch.no_grad():  # Keine Gradient-Berechnung nötig
        for i, (images, _) in enumerate(test_loader):
            grayscale_images = transforms.functional.rgb_to_grayscale(images).to(device)
            images = images.to(device)
            outputs = model(grayscale_images)
            outputs = (outputs * 255).to(torch.uint8)

            # Plotte das Original, das Grauwertbild und das eingefärbte Bild
            plot_images(images, grayscale_images, outputs)

            break  # Nur für ein Beispiel, damit die Schleife nicht endlos läuft"""
           

