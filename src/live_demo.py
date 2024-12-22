import cv2 as cv
import torch
import os
#from network import Net
#from transforms import ValidationTransform
from PIL import Image
import matplotlib.pyplot as plt
from plotting_results import colorization
from Imagenette_colorization import transform, im_convert
import torchvision
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def live(args):
    if args.dataset == "Cifar10":
        from CIFAR_colorization import ConvNet
        trained_model_path = "./models_Cifar10/model_20241104_232646_18"
            
    elif args.dataset == "Imagenette":
        from Imagenette_colorization import ConvNet
        #trained_model_path = ".\models_Imagenette\model_20241126_173658_10"
        #trained_model_path = ".\models_Imagenette\model_20241205_232140_0"
        trained_model_path = ".\models_Imagenette\model_20241209_005507_0"
        print("Imagenette")


    # Load the model checkpoint from a previous training session (check code in train.py)
    #checkpoint = torch.load(trained_model_path)
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(trained_model_path, weights_only="True"))
    model.eval()
    

    # Create a video capture device to retrieve live footage from the webcam.
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Webcam did not open")
        

    # Calculate the border sizes once
    ret, frame = cap.read()
    if not ret:
        raise IOError("No Image found")

    #resizing to 320,320 (model-input size)
    frame = frame.resize((320,320))

    # for continuous processing of the frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            raise IOError("No Image found")
        
        frame = cv.resize(frame, (320, 320))

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #converting np array to PIL to be able to apply transforms
        frame_PIL = Image.fromarray(np.uint8(frame))
        #applying transforms
        frame_transformed = transform(frame_PIL)
        #image to grayscale
        grayscale_images = torchvision.transforms.functional.rgb_to_grayscale(frame_transformed).to(device)
        #adding batch
        grayscale_images = grayscale_images.unsqueeze(0) #mein Model erwartet input in form [b,c,w,h]
        #colorizing with model
        outputs = model(grayscale_images)
        #batch tensor to np array
        colorized_image = im_convert(outputs[0])#im_convert takes batches

        cv.imshow('Input',frame)
        cv.imshow('grayscale', gray)
        cv.imshow('colorized', colorized_image)

        c = cv.waitKey(1)
        if c == 10:
            break
       

    # Release the video capture resources and close all OpenCV windows.
    cap.release()
    cv.destroyAllWindows()