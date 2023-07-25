import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import argparse
import os

def sliding_window_average_x(x, size=10):
	if(size < 1):
		size = 1
	window_x = []
	avg_x = []
	for x_i in x:
		window_x.append(x_i)
		window_x = window_x[-size:]
		avg_x.append(np.mean(window_x))
	return avg_x

parser = argparse.ArgumentParser(description='Run the model on all the image files in a passed folder. Write the images that have predicted bounding boxes to the output folder for inspection.')
parser.add_argument('--in_metrics_file', type=str, help='The path to the metrics file that needs to be plotted.')
parser.add_argument('--out_plot_path', type=str, help='The path to the folder where all the plots should be saved to.')
parser.add_argument('--moving_average_window', type=int, help='The size of the window from which moving averages will be computed.', default=10)
args = parser.parse_args()

data = pd.read_csv(args.in_metrics_file, header=0)

timestamp_prefix = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

'''
train_val_box_loss = data[["train_box_loss_epoch", "valid_box_loss_epoch"]]
train_box_loss = train_val_box_loss["train_box_loss_epoch"].dropna()
valid_box_loss = train_val_box_loss["valid_box_loss_epoch"].dropna()
train_box_loss_window = sliding_window_average_x(train_box_loss, args.moving_average_window)
valid_box_loss_window = sliding_window_average_x(valid_box_loss, args.moving_average_window)

plt.plot(range(0, len(train_box_loss)), train_box_loss, alpha=0.3, color="C0")
plt.plot(range(0, len(valid_box_loss)), valid_box_loss, alpha=0.3, color="C1")
plt.plot(range(0, len(train_box_loss_window)), train_box_loss_window, label="Train Box Loss", alpha=1, color="C0")
plt.plot(range(0, len(valid_box_loss_window)), valid_box_loss_window, label="Valid Box Loss", alpha=1, color="C1")
plt.yscale("log")
plt.legend()
plt.grid(alpha=0.2)
plt.title("Box Loss vs Epoch\n" + args.in_metrics_file)
plt.savefig(os.path.join(args.out_plot_path, timestamp_prefix + "_box_loss.png"))
plt.clf()
plt.close()

train_val_class_loss = data[["train_class_loss_epoch", "valid_class_loss_epoch"]]
train_class_loss = train_val_class_loss["train_class_loss_epoch"].dropna()
valid_class_loss = train_val_class_loss["valid_class_loss_epoch"].dropna()
train_class_loss_window = sliding_window_average_x(train_class_loss, args.moving_average_window)
valid_class_loss_window = sliding_window_average_x(valid_class_loss, args.moving_average_window)

plt.plot(range(0, len(train_class_loss)), train_class_loss, alpha=0.3, color="C0")
plt.plot(range(0, len(valid_class_loss)), valid_class_loss, alpha=0.3, color="C1")
plt.plot(range(0, len(train_class_loss_window)), train_class_loss_window, label="Train Class Loss", alpha=1, color="C0")
plt.plot(range(0, len(valid_class_loss_window)), valid_class_loss_window, label="Valid Class Loss", alpha=1, color="C1")
plt.yscale("log")
plt.legend()
plt.grid(alpha=0.2)
plt.title("Class Loss vs Epoch\n" + args.in_metrics_file)
plt.savefig(os.path.join(args.out_plot_path, timestamp_prefix + "_class_loss.png"))
plt.clf()
plt.close()
'''

train_val_loss = data[["train_loss_epoch", "valid_loss_epoch"]]
train_loss = train_val_loss["train_loss_epoch"].dropna()
valid_loss = train_val_loss["valid_loss_epoch"].dropna()
train_loss_window = sliding_window_average_x(train_loss, args.moving_average_window)
valid_loss_window = sliding_window_average_x(valid_loss, args.moving_average_window)

#plt.plot(range(0, len(train_loss)), train_loss, alpha=0.3, color="C0")
plt.plot(range(0, len(valid_loss)), valid_loss, alpha=0.3, color="C1")
#plt.plot(range(0, len(train_loss_window)), train_loss_window, label="Train Loss", alpha=1, color="C0")
plt.plot(range(0, len(valid_loss_window)), valid_loss_window, label="Valid Loss", alpha=1, color="C1")
plt.yscale("log")
plt.legend()
plt.grid(alpha=0.2)
plt.title("Loss vs Epoch\n" + args.in_metrics_file)
plt.savefig(os.path.join(args.out_plot_path, timestamp_prefix + "_loss.png"))
plt.clf()
plt.close()