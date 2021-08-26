import pandas as pd
import numpy as np

def get_data(path, seed=0):
	df = pd.read_csv(path)
	df = df.values
	np.random.seed(seed)
	np.random.shuffle(df)
	df_test = df[int(0.75*len(df)):]
	df_test = df_test[np.argsort(df_test.T[0])]
	df = df[:int(0.75*len(df))]
	df = df[np.argsort(df.T[0])]
	labels = df.T[0]
	images = df.T[1:].T
	images = np.reshape(images, (len(df), -1))
	return {'labels': labels, 'images': images, 'test_labels': df_test.T[0], 'test_images': df_test.T[1:].T}