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


parser = argparse.ArgumentParser(
    prog="Grayscale image colorization",
    description="Set mode to either 'training' or 'colorization'. Specify the Dataset.",
    epilog="Students project",
)


parser.add_argument(
    "-m",
    "--mode",
    choices=["training", "colorization"],
    help="Choose which mode you want to execute",
)
#parser.add_argument("-d", "--Dataset", help="Path to Dataset")
parser.add_argument(
    "-d",
    "--dataset",
    choices=["Cifar10", "Imagenette"],
    help="choose which Dataset you want to use to colorize the uploaded images",
)

parser.add_argument(
    "-pl",
    "--plot_examples",
    #nargs="+",
    choices=["True", "False"],
    help="choose wether to plot examples after finishing training",
)

args = parser.parse_args()

# switch control flow based on arguments
if args.mode == "training" and args.dataset == "Cifar10":
    print("training image colorization network...")
    CIFAR_colorization.trainConvNet()
    if args.plot_examples == "True":
        print("plotting examples")
        CIFAR_colorization.plot_examples()
if args.mode == "training" and args.dataset == "Imagenette":
    Imagenette_colorization.trainConvNet()
    if args.plot_examples == "True":
        print("plotting examples")
        Imagenette_colorization.plot_examples()


elif args.mode == "colorization" and args.dataset != None:
    print("coloring example image...")
    #plotting_results.eval_model_and_plot()
    plotting_results.colorization(dataset = args.dataset)