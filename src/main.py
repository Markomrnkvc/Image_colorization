# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 02:57:50 2024

@author: marko
"""
import argparse
import os
#import pandas as pd
import CIFAR_colorization
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
    "--Dataset",
    nargs="+",
    choices=["Cifar10", "Private"],
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
if args.mode == "training":
    print("training image colorization network...")
    CIFAR_colorization.trainConvNet()
    if args.plot_examples == "True":
        print("plotting examples")
        CIFAR_colorization.plot_examples()


elif args.mode == "colorization":
    print("clustering the dataset...")
    plotting_results.eval_model_and_plot()
"""
elif args.mode == "recommender" and args.method != None:
    print("starting recommendation app...")
    recommender = Recommender(methods=args.method)
    recommender.recommend()

elif args.mode == "recommender_no_cluster":
    print("start with the Recommender (no clusters)")
    recommender_no_cluster = Recommender_NC(methods=args.method)
    recommender_no_cluster.recommend()"""