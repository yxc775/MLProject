from keras.models import Sequential
from keras.optimizers import Adam
from GAN import generator, discriminator


def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(lr=0.0002,beta_1=0.5)
    model.compile(loss='binary_crossentropy',optimizer=opt)
    return model


if __name__ == '__main__':
    # size of the latent space
    latent_dim = 100
    # create the discriminator
    d_model = discriminator.define_model()
    # create the generator
    g_model = generator.define_model(latent_dim)
    # create the gan
    gan_model = define_gan(g_model, d_model)
    gan_model.summary()
