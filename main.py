
#Convolutional Autoencoder

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import  keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
import numpy as np

def import_data():
    (train_images, train_labels), (test_images, test_labels)=keras.datasets.mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)
def show_images(number, test_images):

    for i in range(number):
        plt.subplot(2, number, i + 1)
        plt.imshow(test_images[i].reshape(28, 28))
        plt.gray()
        plt.show()
def add_noise(x_train, x_test):
    noise_factor = 0.4
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return (x_train_noisy, x_test_noisy )
def show_clean_noisy(images_noisy,images_clean):

    n = 10
    plt.figure(figsize=(20, 10))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images_noisy[i].reshape(28, 28))
        plt.gray()

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(images_clean[i].reshape(28, 28))
        plt.gray()
    plt.show()




def main():

    (train_images, train_labels), (test_images, test_labels)= import_data()
    print("========>Data imported!")


    train_images=train_images.astype('float32')/255
    test_images = test_images.astype('float32') / 255

    train_images=np.reshape(train_images,(len(train_images),28,28,1))
    test_images=np.reshape(test_images,(len(test_images), 28, 28, 1))

    (train_images_noisy, test_images_noisy) = add_noise(train_images, test_images)

    show_clean_noisy(test_images_noisy,test_images)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D((2, 2), padding='same'))

    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.summary()

    model.fit(train_images_noisy, train_images, epochs=10, batch_size=256, shuffle=True,
              validation_data=(test_images_noisy, test_images))

    decoded_images = model.predict(test_images_noisy)

    n = 10
    plt.figure(figsize=(20, 10))
    for i in range(n):

        plt.subplot(2, n, i + 1)
        plt.imshow(test_images_noisy[i].reshape(28, 28), cmap="binary")

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_images[i].reshape(28, 28), cmap="binary")

    plt.show()



if __name__ == "__main__":
    main()

