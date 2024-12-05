# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 02:57:50 2024

@author: marko
"""
import argparse
import os
import torch
import CIFAR_colorization
import ImageNet_colorization
import plotting_results

# picklefiley = "/Users/mjy/Downloads/data_clustered_5kentries.pkl"
# data = pd.read_pickle(picklefiley)
# print(data.head())

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Setup an argument parser for control via command line
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
    choices=["Cifar10", "ImageNet"],
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
if args.mode == "training" and args.dataset == "ImageNet":
    ImageNet_colorization.trainConvNet()
    if args.plot_examples == "True":
        print("plotting examples")
        ImageNet_colorization.plot_examples()


elif args.mode == "colorization" and args.dataset != None:
    print("coloring example image...")
    #plotting_results.eval_model_and_plot()
    plotting_results.colorization(dataset = args.dataset)
"""
elif args.mode == "recommender" and args.method != None:
    print("starting recommendation app...")
    recommender = Recommender(methods=args.method)
    recommender.recommend()

elif args.mode == "recommender_no_cluster":
    print("start with the Recommender (no clusters)")
    recommender_no_cluster = Recommender_NC(methods=args.method)
    recommender_no_cluster.recommend()"""