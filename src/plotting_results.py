import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
#from CIFAR_colorization import ConvNet, rgb_to_gray, test_dataset, test_loader, train_loader, train_dataset
from ImageNet_colorization import ConvNet, rgb_to_gray, test_dataset, test_loader, train_loader, train_dataset
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def colorization(dataset):
    
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    images_per_row =5
    num_images = 30
    def im_convert(tensor):
        image = tensor.cpu().clone().detach().numpy()
        image = np.transpose(image, (1, 2, 0))  # Von (C, H, W) zu (H, W, C)
        return image

    # Visualisierungsfunktion für Bilder in mehreren Reihen mit maximal 5 Bildern pro Reihe
    def plot_images_grid(originals, grayscales, colorized_images, images_per_row=images_per_row):
        num_images = min(len(originals), len(grayscales), len(colorized_images))
        num_rows = (num_images + images_per_row - 1) // images_per_row  # Berechnet die Anzahl der benötigten Reihen

        fig, axes = plt.subplots(num_rows, images_per_row * 3, figsize=(18, 6 * num_rows))

        for idx in range(num_images):
            row = idx // images_per_row
            col = (idx % images_per_row) * 3

            # Originalbild
            original_img = im_convert(originals[idx])
            axes[row, col].imshow(original_img)
            axes[row, col].axis('off')

            # Grauwertbild
            grayscale_img = im_convert(grayscales[idx])
            axes[row, col + 1].imshow(grayscale_img, cmap='gray')
            axes[row, col + 1].axis('off')

            # Koloriertes Bild
            colorized_img = im_convert(colorized_images[idx])
            axes[row, col + 2].imshow(colorized_img)
            axes[row, col + 2].axis('off')

        # Entferne Abstände zwischen den Subplots
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def denormalize(tensor):
        denorm = tensor.clone()
        for t in range(denorm.size(0)):
            denorm[t] = denorm[t] * 0.5 + 0.5
        return denorm

    def rgb_to_gray(img):
        return img.mean(dim=1, keepdim=True)

    def eval_model_and_plot(num_images=num_images):
        try:
            model.eval()
            
            optimizer = optim.Adam(model.parameters(), lr = 0.005)
        except (NameError, AttributeError):
            model = ConvNet().to(device)
            optimizer = optim.Adam(model.parameters(), lr = 0.005)
            model.load_state_dict(torch.load(trained_model_path, weights_only="True"))

            #model.load_state_dict(torch.load(trained_model_path, map_location=device), strict= True)


            #checkpoint = torch.load(trained_model_path, weights_only=True, map_location=device)
            #model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            
            """
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    print(f"{name}: running_mean={module.running_mean}, running_var={module.running_var}")
            
            
            model.train()  # Setzen Sie das Modell in den Trainingsmodus# BatchNorm mit Trainingsdaten kalibrieren
            with torch.no_grad():
                for inputs, _ in train_loader:  # Oder ein ähnlicher Daten-Loader
                    inputs = rgb_to_gray(inputs).to(device)
                    model(inputs)
            #
            """
            model.eval().to(device)

        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                grayscale_images = rgb_to_gray(images).to(device)
                images = images.to(device)
                #images = invTrans(images)
                outputs = model(grayscale_images)
                
                
                # Zeige die ersten 'num_images' Bilder an, jeweils 5 pro Reihe
                plot_images_grid(images[:num_images], grayscale_images[:num_images], outputs[:num_images], images_per_row=5)
                break  # Nur eine Batch anzeigen
    
    if dataset == "Cifar10":
        from CIFAR_colorization import ConvNet, rgb_to_gray, test_dataset, test_loader, train_loader, train_dataset
        trained_model_path = "./models/model_20241104_232646_18"
            
    elif dataset == "ImageNet":
        from ImageNet_colorization import ConvNet, rgb_to_gray, test_dataset, test_loader, train_loader, train_dataset
        trained_model_path = ".\models\model_20241126_173658_10"

    eval_model_and_plot(num_images)