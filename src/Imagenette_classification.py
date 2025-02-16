import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from classification_dataset_Imagenette import PixelClassImagenette
from skimage.color import rgb2lab, lab2rgb

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
start_epoch = 0
EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 0.0005
NUM_CLASSES = 256  # 256 Klassen pro Kanal (weil 256 RGB-Werte)

# Transform
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)

try:
    train_dataset = PixelClassImagenette(
        root="./data_Imagenette", train=True, transform=transform, download=True
    )
    test_dataset = PixelClassImagenette(
        root="./data_Imagenette", train=False, transform=transform, download=True
    )

except RuntimeError:
    train_dataset = PixelClassImagenette(
        root="./data_Imagenette", train=True, transform=transform, download=False
    )
    test_dataset = PixelClassImagenette(
        root="./data_Imagenette", train=False, transform=transform, download=False
    )

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Model definition
class ConvNet(nn.Module):
    def __init__(self, num_classes=256):
        super(ConvNet, self).__init__()
        self.batnchnorm16 = nn.BatchNorm2d(16)#batchnormalization for input layer 
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1, stride=2)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)  
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)  
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2) 

        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')  
        self.conv6 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')  
        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear')  
        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear') 
        self.conv9 = nn.Conv2d(64, num_classes * 3, kernel_size=3, padding=1)  # 3 Kanäle × 256 Klassen

    def forward(self, x):
        # Encoder
        x = self.batnchnorm16(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))  
        x = F.relu(self.conv3(x))  
        x = F.relu(self.conv4(x))  
        x = F.relu(self.conv5(x))  

        # Decoder
        x = self.upsample1(x)  
        x = F.relu(self.conv6(x))
        x = self.upsample2(x) 
        x = F.relu(self.conv7(x))
        x = self.upsample3(x)  
        x = F.relu(self.conv8(x))
        x = self.upsample4(x)  
        x = self.conv9(x)  # Kein Sigmoid, weil Klassifikation
        return x


model = ConvNet(num_classes=NUM_CLASSES).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

runs_path = None  # "Imagenette_pixel_classification_lab_20250109_024009"#Imagenette_pixel_classification_lab_20250104_132551"


# trained_model = "./models_Imagenette_classification_lab/model_20250109_024009_0"
# start_epoch = 0
# checkpoint = torch.load(trained_model)
# start_epoch = checkpoint['epoch']+1
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])
def compute_loss(outputs, targets):
    red_logits, green_logits, blue_logits = torch.chunk(
        outputs, chunks=3, dim=1
    )  # teilt torch.Size([36, 768, 32, 32]) in 3X torch.Size([36, 256, 32, 32]) auf

    red_targets = targets[:, 0, :, :]  # weil Rot-Wert dder pixel hier gespeichert
    green_targets = targets[:, 1, :, :]  # weil Grün-Wert dder pixel hier gespeichert
    blue_targets = targets[:, 2, :, :]  # weil Blau-Wert dder pixel hier gespeichert

    # computing CrossEntropyLoss for all colors
    red_loss = criterion(red_logits, red_targets)
    green_loss = criterion(green_logits, green_targets)
    blue_loss = criterion(blue_logits, blue_targets)

    return (red_loss + green_loss + blue_loss) / 3


# Training and validation loop
# model = ConvNet(num_classes=NUM_CLASSES).to(device)
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch(epoch_index, writer):
    running_loss = 0.0
    for i, (images, labels) in tqdm(
        enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch_index}"
    ):
        grayscale_images = transforms.functional.rgb_to_grayscale(images).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(grayscale_images)

        loss = compute_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        # print(labels.shape)
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar("Loss/train", avg_loss, epoch_index)
    return avg_loss


def validation_loss(epoch_index, writer):
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, (vimages, vlabels) in enumerate(test_loader):
            grayscale_images = transforms.functional.rgb_to_grayscale(vimages).to(
                device
            )
            vlabels = vlabels.to(device)
            voutputs = model(grayscale_images)
            vloss = compute_loss(voutputs, vlabels)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / len(test_loader)
    writer.add_scalar("Loss/validate", avg_vloss, epoch_index)
    return avg_vloss


# Training the model
def train_model():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if runs_path != None:
        writer = SummaryWriter("runs_lab/{}".format(runs_path))
    else:
        writer = SummaryWriter(
            "runs_lab/Imagenette_pixel_classification_lab_{}".format(timestamp)
        )

    best_vloss = float("inf")

    for epoch in tqdm(
        range(start_epoch, EPOCHS),
        desc="Progress for complete training (percentage of finished EPOCHS)",
    ):
        print("EPOCH {}:".format(epoch))

        model.train(True)
        avg_loss = train_one_epoch(epoch, writer)
        avg_vloss = validation_loss(epoch, writer)

        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = "./models_Imagenette_classification/model_{}_{}".format(
                timestamp, epoch
            )
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, model_path)
        elif epoch == EPOCHS:
            model_path = "./models_Imagenette_classification_lab/model_{}_{}".format(
                timestamp, epoch
            )

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, model_path)
            # torch.save(model.state_dict(), model_path)
            print(f"saved checkpoint in {model_path}")


# Visualization
def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    return image


def plot_images(original, grayscale, colorized):
    original_img = im_convert(original[0])
    # print(grayscale.shape)
    grayscale_img = grayscale[0].cpu().numpy()[0]

    # print(grayscale_img.shape)
    colorized_img = im_convert(colorized[0])
    # print(colorized_img.shape)
    fig, ax = plt.subplots(1, 4, figsize=(24, 8))
    lab_colorized = rgb2lab(colorized_img / 255)
    lab_original = rgb2lab(original_img)

    l_original, a_original, b_original = extract_channels_plt(lab_original)
    l_colorized, a_colorized, b_colorized = extract_channels_plt(lab_colorized)
    colorized_lab = np.stack([l_original, a_colorized, b_colorized], axis=-1)
    colorized_lab = lab2rgb(colorized_lab)
    ax[0].imshow(original_img)
    ax[0].set_title("Original RGB Image")
    ax[0].axis("off")
    ax[1].imshow(grayscale_img, cmap="gray")
    ax[1].set_title("Grayscale Image")
    ax[1].axis("off")
    ax[2].imshow(colorized_img)
    ax[2].set_title("Colorized by Model")
    ax[3].axis("off")
    ax[3].imshow(colorized_lab)
    ax[3].set_title("Colorized by Model LAB")
    ax[3].axis("off")
    plt.show()


def plot_examples(model=model):
    model.eval().to(device)
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            grayscale_images = transforms.functional.rgb_to_grayscale(images).to(device)
            images = images.to(device)
            outputs = model(grayscale_images)
            red_logits, green_logits, blue_logits = torch.chunk(
                outputs, chunks=3, dim=1
            )
            red_classes = red_logits.argmax(dim=1)
            green_classes = green_logits.argmax(dim=1)
            blue_classes = blue_logits.argmax(dim=1)
            colorized_output = torch.stack(
                [red_classes, green_classes, blue_classes], dim=1
            )
            plot_images(images, grayscale_images, colorized_output)
            break


def extract_channels_plt(lab_image):
    # Sicherstellen, dass das LAB-Bild ein Tensor bleibt und auf der CPU verarbeitet wird
    # lab_image = lab_image.cpu().clone().detach()  # Auf CPU verschieben und keine Gradienten mehr verfolgen

    # Extrahiere die Kanäle für die gesamte Batch
    l = lab_image[:, :, 0]  # L-Kanal, erhaltene Form: [Batchsize, 1, H, W]
    a = lab_image[:, :, 1]  # a-Kanal, erhaltene Form: [Batchsize, 1, H, W]
    b = lab_image[:, :, 2]  # b-Kanal, erhaltene Form: [Batchsize, 1, H, W]
    return l, a, b
