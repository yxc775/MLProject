from keras.models import Sequential
from keras.optimizers import SGD


def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = SGD(lr=0.0002, momentum=0.005)
    model.compile(loss='binary_crossentropy',optimizer=opt)
    return model

