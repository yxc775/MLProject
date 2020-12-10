import numpy as np
# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import uniform
from matplotlib import pyplot


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()


def createNumPair():
	list = []
	for x in range(25):
		pair = [uniform(-2.0,2.0),uniform(-2.0,2.0)]
		list.append(pair)

	return np.array(list)

if __name__ == '__main__':
	# load model
	model = load_model('decoderModel.h5')
	# generate images
	latent_points = createNumPair()
	# generate image
	X = model.predict(latent_points)
	# plot the result
	save_plot(X, 5)
