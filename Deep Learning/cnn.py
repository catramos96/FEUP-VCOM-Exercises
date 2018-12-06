from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

def main():
    ## load dataset


    ## create model


    ## train the model


    ## test the model


def loadMNISTdataset():
    from keras.datasets import mnist
    img_w = img_h = 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_dim_ordering() == 'th':
        x_train = x_train.reshape(x_train.shape[0], 1, img_w, img_h)
        x_test = x_test.reshape(x_test.shape[0], 1, img_w, img_h)
        input_shape = (1, img_w, img_h)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_w, img_h, 1)
        x_test = x_test.reshape(x_test.shape[0], img_w, img_h, 1)
        input_shape = (img_w, img_h, 1)
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    return (x_train, y_train, x_test, y_test, input_shape)


if __name__=='__main__':
    main()
