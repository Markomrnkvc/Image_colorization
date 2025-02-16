# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 02:57:50 2024

@author: marko
"""
import argparse
import os
import torch
import CIFAR_colorization
import Imagenette_colorization
import plotting_results
import live_demo
import CIFAR10_classification
import classification_plot_examples


parser = argparse.ArgumentParser(
    prog="Grayscale image colorization",
    description="Set mode to either 'training' or 'colorization'. Specify the Dataset.",
    epilog="Students project",
)


parser.add_argument(
    "-m",
    "--mode",
    choices=["training", "colorization", "live"],
    help="Choose which mode you want to execute",
)
parser.add_argument(
    "-d",
    "--dataset",
    choices=["Cifar10", "Imagenette"],
    help="choose which Dataset you want to use to colorize the uploaded images",
)

parser.add_argument(
    "-pr",
    "--problem",
    #nargs="+",
    choices=["regression", "classification"],
    default = "classification",
    help="choose wether you want to train the regression model or the classification model",
)

parser.add_argument(
    "-pl",
    "--plot_examples",
    #nargs="+",
    choices=["True", "False"],
    default = "True",
    help="choose wether to plot examples after finishing training",
)


args = parser.parse_args()

# switch control flow based on arguments
if args.mode == "training" and args.dataset == "Cifar10":
    if args.problem == "regression":
        print("training regression image colorization network...\n")
        CIFAR_colorization.trainConvNet()
        if args.plot_examples == "True":
            print("plotting examples")
            CIFAR_colorization.plot_examples()

    elif args.problem == "classification":
        print("training classification image colorization network...\n")
        CIFAR10_classification.train_model()
        if args.plot_examples == "True":
            print("plotting examples")
            CIFAR10_classification.plot_examples()

elif args.mode == "training" and args.dataset == "Imagenette":
    Imagenette_colorization.trainConvNet()
    if args.plot_examples == "True":
        print("plotting examples\n")
        Imagenette_colorization.plot_examples()

elif args.mode == "colorization" and args.dataset == "Cifar10":
    if args.problem == "regression":
        print("coloring example image...\n")
        plotting_results.colorization(dataset = args.dataset)
    elif args.problem == "classification":
        classification_plot_examples.plot_examples()

elif args.mode == "live" and args.dataset == "Imagenette":
    print("coloring example image...\n")
    #plotting_results.eval_model_and_plot()
    live_demo.live(args)    
    
"""
elif args.mode == "colorization" and args.dataset != None:
    print("coloring example image...")
    #plotting_results.eval_model_and_plot()
    plotting_results.colorization(dataset = args.dataset)
"""


