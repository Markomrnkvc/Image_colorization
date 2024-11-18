import torch
import matplotlib.pyplot as plt
import numpy as np
from CIFAR_colorization import ConvNet, rgb_to_gray, test_dataset, test_loader, train_dataset, train_loader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    #print(colorized_img.mean())
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

def denormalize(tensor):
    # Annahme: tensor hat Form (C, H, W) und ist im Bereich [-1, 1] nach Normalisierung
    denorm = tensor.clone()  # Erstellen einer Kopie des Tensors, um das Original zu bewahren
    for t in range(denorm.size(0)):  # Für jeden Farbkanal
        denorm[t] = denorm[t] * 0.5 + 0.5  # Rücknormalisierung mit Standardabweichung und Mittelwert
    return denorm

def eval_model_and_plot():
    # Beispiel: Zeige ein Bild, graues Bild und das durch das Modell eingefärbte Bild an
    try:
        model.eval()  # Schalte das Modell in den Evaluierungsmodus
    except (NameError,AttributeError) as e:
        #loading model if not loaded already
        #need to adjust the path manually
        trained_model_path = ".\models\model_20241118_234506_0"
        model = ConvNet()
        model.load_state_dict(torch.load(trained_model_path, weights_only="True"))
        model.eval().to(device)

    #if images != None:
    with torch.no_grad():  # Keine Gradient-Berechnung nötig
        for i, (images, _) in enumerate(train_loader):
            #images = denormalize(images)
            grayscale_images = rgb_to_gray(images).to(device)
            images = images.to(device)
            # Vorhersage durch das Modell
            outputs = model(grayscale_images)

            
            #outputs = denormalize(outputs)
            # Plotte das Original, das Grauwertbild und das eingefärbte Bild
            plot_images(images, grayscale_images, outputs)

            break  # Nur für ein Beispiel, damit die Schleife nicht endlos läuft"""
