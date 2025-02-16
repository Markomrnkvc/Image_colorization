import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def colorize(args):

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
            
    if args.dataset == "Cifar10":
        from CIFAR_colorization import ConvNet, test_loader
        trained_model_path = "./models_Cifar10/model_20250216_115232_79"
    elif args.dataset == "Imagenette":
        from Imagenette_colorization import ConvNet, test_loader
        trained_model_path = ".\models_Imagenette\model_20241126_173658_10"

    eval_model_and_plot()