from keras.models import Sequential
from keras.optimizers import Adam


def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(lr=0.0002,beta_1=0.05)
    model.compile(loss='binary_crossentropy',optimizer=opt)
    return model

