# example of loading the mnist dataset
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import ReLU
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import numpy.random as rd
from matplotlib import pyplot


def showMinist():
    # load the images into memory
    (trainX, trainy), (testX, testy) = load_data()
    # summarize the shape of the dataset
    print('Train', trainX.shape, trainy.shape)
    print('Test', testX.shape, testy.shape)
    for i in range(400):
        # define subplot
        pyplot.subplot(20, 20, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(trainX[i], cmap='gray_r')
    pyplot.show()


def define_model(in_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (2, 2), strides=(1, 1), padding='same'))
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.05)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def load_minist():
    (trainX, _), (_, _) = load_data()
    # expand to 3d, e.g. add channels dimension
    X = np.expand_dims(trainX, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [0,1]
    X = X / 255.0
    return X


def pick_real_sample(dataset, n):
    ix = rd.randint(0, dataset.shape[0], n)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = np.ones((n, 1))
    return X, y


def create_fake_sample(n):
    # generate uniform random numbers in [0,1]
    X = rd.rand(28 * 28 * n)
    # reshape into a batch of grayscale images
    X = X.reshape((n, 28, 28, 1))
    # generate 'fake' class labels (0)
    y = np.zeros((n, 1))
    return X, y


def train_discriminator_model(model,dataset,n_iter=100,n_batch=256):
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_iter):
        # get randomly selected 'real' samples
        X_real, y_real = pick_real_sample(dataset, half_batch)
        # update discriminator on real samples
        _, real_acc = model.train_on_batch(X_real, y_real)
        # generate 'fake' examples
        X_fake, y_fake = create_fake_sample(half_batch)
        # update discriminator on fake samples
        _, fake_acc = model.train_on_batch(X_fake, y_fake)
        # summarize performance
        print('>%d real=%.0f%% fake=%.0f%%' % (i + 1, real_acc * 100, fake_acc * 100))


if __name__ == '__main__':
    # define model
    model = define_model()
    # summarize the model
    model.summary()
