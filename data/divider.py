import pandas as pd
import numpy as np
import cv2
from tqdm import trange

def main():
	df = pd.read_csv("mnist.csv")
	df = df.values
	labels = df.T[0]
	df = df.T[1:].T
	for label in trange(10):
		indexes = np.argwhere(labels==label)
		x = df[indexes]
		np.random.shuffle(x)
		x = x[:100]
		x = x.reshape((len(x), 28, 28))
		for idx, image in enumerate(x):
			cv2.imwrite('./MNIST/'+str(label)+'/'+"0"*(3-len(str(idx)))+str(idx)+".png", image)

if __name__ == '__main__':
	main()