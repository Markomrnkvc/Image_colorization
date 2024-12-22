import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import cv2 as cv

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

            #outputs_array = outputs.to("cpu")
            #outputs_array = outputs_array.numpy()
            hsv = cv.cvtColor(colorized_img, cv.COLOR_BGR2HSV)
            if idx == 1:
                print(hsv[:,:,1])
                print(hsv[0,0,1])
                print("------------------------")

                

            #print(hsv[0].mean())
            #hsv[:,:,2] = hsv[:,:,2]*1.5
            #hsv[:,:,1] = hsv[:,:,1]*1.5
            hsv[:,:,0] = hsv[:,:,0]*1.5
            if idx == 1:
                print(hsv[:,:,1])
                print(hsv[0,0,1])
                print("------------------------")
                print(hsv[:][:][1].shape)
                print(hsv[:][:][:][:][:][1])
                print(hsv.shape)

                #print(hsv)
            #(1 - kontrast)*
            #print(hsv[0].shape)
            rgb = cv.cvtColor(colorized_img, cv.COLOR_HSV2BGR)
            rgb = cv.cvtColor(colorized_img, cv.COLOR_BGR2RGB)
            
            axes[row, col + 2].imshow(rgb)
            axes[row, col + 2].axis('off')

        # Entferne Abstände zwischen den Subplots
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    def eval_model_and_plot(num_images=num_images):
        try:
            model.eval()
            
            optimizer = optim.Adam(model.parameters(), lr = 0.005)
        except (NameError, AttributeError):
            model = ConvNet().to(device)
            optimizer = optim.Adam(model.parameters(), lr = 0.005)
            model.load_state_dict(torch.load(trained_model_path, weights_only="True"))

            model.eval().to(device)

        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                #grayscale_images = rgb_to_gray(images).to(device)
                grayscale_images = transforms.functional.rgb_to_grayscale(images).to(device)
                images = images.to(device)
                outputs = model(grayscale_images)
                if dataset == "Cifar10":
                    outputs = (outputs * 255).to(torch.uint8)

                
                
                # Zeige die ersten 'num_images' Bilder an, jeweils 5 pro Reihe
                plot_images_grid(images[:num_images], grayscale_images[:num_images], outputs[:num_images], images_per_row=images_per_row)
                break  
    
    if dataset == "Cifar10":
        from CIFAR_colorization import ConvNet,  test_dataset, test_loader, train_loader, train_dataset
        #trained_model_path = "./models_Cifar10/model_20241104_232646_18"
        #trained_model_path = "./models_Cifar10/model_20241208_193034_14"
        #trained_model_path = "./models_Cifar10/model_20241208_181658_49" #das hier ist die Unet structure
        trained_model_path = "./models_Cifar10/model_20241211_124608_99"
    elif dataset == "Imagenette":
        from Imagenette_colorization import ConvNet, test_dataset, test_loader, train_loader, train_dataset
        #trained_model_path = ".\models_Imagenette\model_20241126_173658_10"
        trained_model_path = ".\models_Imagenette\model_20241209_005507_7"

    eval_model_and_plot(num_images)