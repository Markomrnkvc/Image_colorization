# Image_colorization
![](https://github.com/Markomrnkvc/Image_colorization/blob/readme/gif_colorization.gif)
---------
#### Install with Anaconda

You should now be able to do a simple install with Anaconda. Here are the steps:

Open the command line and navigate to the folder where you want to download the repo.  Then
type the following commands

```console
git clone https://github.com/Markomrnkvc/Image_colorization.git
cd Image_colorization
conda env create -f environment.yml
```
--> the datasets needed will be downloaded automatically :)

Now you can start using the code! 


#### Using the code

###### for the easiest way to use the code navigate to the /src folder within the repo

The code uses argparse parsing to start different Functions and methods or to choose between the datasets
- "--mode": choose between the modes:
    - training (training the network)
    - colorization (colorizing an example image)
    - live (using colorizing on your webcam)
    - diashow (colorizing images saved on a folder on your PC)
- "--dataset": choose between [CIFAR10,Imagenette]
    - you will need to choose a dataset for the modes: training, colorization, live
- "--problem": choose between the type of problem [regression, classification]
    - the code has different networks which color images using regression or classification (classification works better ;) )
- "--folder": input the folder your images are saved in
    - you will need this argument for mode diashow


#### some examples for starting the code via your terminal
- python main.py --mode "training" --dataset "Cifar10" --problem "regression" (training regression model for CIFAR10)
- python main.py --mode "training" --dataset "Cifar10" --problem "classification" (training classification model for CIFAR10)
- python main.py --mode "colorization" --dataset "Imagenette" --problem "classification" (colorizing an example image from Imagenette dataset's trained classification model)
- python main.py --mode "live" --dataset "Imagenette" (starting live mode based on Imagenette trained classification model)


#### My poster for presenting the Project at our University's AI-CON
![](https://github.com/Markomrnkvc/Image_colorization/blob/readme/AI_Poster_Marinkovic.pdf)
