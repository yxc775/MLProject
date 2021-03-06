# example of defining and using the generator model
from numpy import zeros
from numpy.random import randn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU, ReLU, Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import UpSampling2D
from keras.initializers import  RandomNormal
from matplotlib import pyplot


# define the standalone generator model
def define_model(dim):
    model = Sequential()
    # foundation for 7x7 image
    n_nodes = 256 * 7 * 7
    model.add(Dense(n_nodes, input_dim=dim))
    model.add(BatchNormalization(momentum=0.9))
    model.add(ReLU())
    model.add(Reshape((7, 7, 256)))
    model.add(Dropout(0.4))
    # upsample to 14x14
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(128, 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(ReLU())

    model.add(UpSampling2D())
    model.add(Conv2DTranspose(64, 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(ReLU())
    # upsample to 28x28
    model.add(Conv2DTranspose(32, 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(ReLU())
    #out 28x28x1
    model.add(Conv2DTranspose(1,5,padding='same'))
    model.add(Activation('sigmoid'))
    return model


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y



if __name__ == '__main__':
    # define model
    model = define_model(100)
    # summarize the model
    model.summary()

