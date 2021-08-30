import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from models.RBM.RBM_synthetic_generator import Pre_trainer as Pre_trainer_synthetic_generator
from matplotlib import pyplot as plt
import numpy as np
import torch
import cv2
from tqdm import tqdm

def get_data(path):
	data = []
	for image_path in tqdm([path+i for i in os.listdir(path)]):
		image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
		data.append(image)
	data = np.stack(data)
	return data

def reform_mnist(synthetic_images, hidden_features):
	synthetic_images = np.reshape(synthetic_images, (-1, 28, 28))
	hidden_features = np.reshape(hidden_features, (-1, 5, 2))
	plt.cla()
	fig = plt.figure(figsize = (15,60))
	for digit in range(10):
		index = np.argwhere(labels==digit).flatten()[0]
		plt.subplot(10, 3, digit*3+1)
		plt.axis('off')
		plt.imshow(images[index], cmap='gray')
		plt.subplot(10, 3, digit*3+2)
		plt.axis('off')
		plt.imshow(hidden_features[index], cmap='gray')
		plt.subplot(10, 3, digit*3+3)
		plt.axis('off')
		plt.imshow(synthetic_images[index], cmap='gray')
	fig.tight_layout()
	plt.savefig('./images/RBM_synthetic_generations.png')

def RBM_synthetic_generation_example(images):
	images_reshaped = images.reshape((len(images), -1))
	images_reshaped = (images_reshaped - np.min(images_reshaped))/(np.max(images_reshaped) - np.min(images_reshaped))
	rbm = Pre_trainer_synthetic_generator(n_visible=images_reshaped.shape[1], n_hidden=400*3, epochs=300, optim='adm', k=20)
	rbm.train_rbm(images_reshaped)
	synthetic_images, hidden_features = rbm.get_synthetic_data(images_reshaped)
	return synthetic_images.reshape(images.shape)

def generate_class_wise(path):
	output_dir = path.replace("data", "synthetic_data")
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	for file_path in [path+i for i in os.listdir(path)]:
		output_path = file_path.replace("data", "synthetic_data")
		if not os.path.exists(output_path):
			os.mkdir(output_path)
		data = get_data(file_path+"/")
		synthetic = RBM_synthetic_generation_example(data)
		for idx, image in tqdm(enumerate(synthetic), total=len(synthetic), desc="Saving"):
			save_path = output_path+"/"+"0"*(len(str(len(synthetic)))-len(str(idx)))+str(idx)+".png"
			cv2.imwrite(save_path, image*255)

def generate_complete_dataset(path):
	output_dir = path.replace("data", "synthetic_data/RBM")
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	dataset = None
	classes_lens = {}
	for file_path in [path+i for i in os.listdir(path)]:
		output_path = file_path.replace("data", "synthetic_data/RBM")
		if not os.path.exists(output_path):
			os.mkdir(output_path)
		data = get_data(file_path+"/")
		classes_lens.update({output_path:data.shape[0]})
		if dataset is None:
			dataset = data
		else:
			dataset = np.concatenate((dataset, data), 0)
	synthetic = RBM_synthetic_generation_example(dataset)
	tots = 0
	for output_path, l in classes_lens.items():
		bar = tqdm(enumerate(synthetic[tots:tots+l]), total=l, desc="Saving")
		for idx, image in bar:
			save_path = output_path+"/"+"0"*(len(str(l))-len(str(idx)))+str(idx)+".png"
			cv2.imwrite(save_path, image*255)
		tots += l
		bar.close()

	
if __name__ == '__main__':
	path = "./data/CIFAR/"
	generate_complete_dataset(path)
	