#gelabierto
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np
import os.path
import matplotlib.pyplot as plt
import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, UpSampling2D, BatchNormalization, Input
import pickle
#References
#https://medium.com/analytics-vidhya/denoising-autoencoder-on-colored-images-using-tensorflow-17bf63e19dad
#https://books.google.es/books?hl=es&lr=&id=20EwDwAAQBAJ&oi=fnd&pg=PP1&dq=all+activation+functions+keras&ots=lHhEakaTU1&sig=-DITKpvlfsLfExXtHl-_FMwF5i4#v=onepage&q=all%20activation%20functions%20keras&f=false
#https://github.com/wbhu/DnCNN-tensorflow

#Load and return the complete data
def load_dataset():
    IMG_SIZE = 100
    dataset = []
    directory = r"/home/marti/Desktop/Datasets/Catsvsdogs/test"
    counter = 0;
    for img in os.listdir(directory):
        try:
            array_img = cv2.imread(os.path.join(directory, img))
            new_img_array = cv2.resize(array_img, (IMG_SIZE, IMG_SIZE))
            #new_img_array=cv2.cvtColor(new_img_array, cv2.COLOR_BGR2RGB)
            #plt.imshow(new_img_array)
            #plt.show()
            dataset.append(new_img_array)
            print("Load of ", img, " completed")
            counter = counter + 1
            if counter == 2000:
                break
        except Exception as error:
            pass

    return dataset

def create_model():

    x = Input(shape=(100, 100, 3))  # Encoder

    e_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    pool1 = MaxPooling2D((2, 2), padding='same')(e_conv1)

    #BatchNormalization used to normalize the input layer by re-centering and re-scaling.
    batchnorm_1 = BatchNormalization()(pool1)
    e_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(batchnorm_1)
    pool2 = MaxPooling2D((2, 2), padding='same')(e_conv2)
    batchnorm_2 = BatchNormalization()(pool2)
    e_conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(batchnorm_2)
    h = MaxPooling2D((2, 2), padding='same')(e_conv3)  # Decoder
    d_conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
    up1 = UpSampling2D((2, 2))(d_conv1)
    d_conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(d_conv2)
    d_conv3 = Conv2D(16, (3, 3), activation='relu')(up2)
    up3 = UpSampling2D((2, 2))(d_conv3)
    r = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)
    model = keras.Model(x, r)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model

def show_images(n, img_noisy, decoded_imgs):


    for i in range(n):


        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(img_noisy[i])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('Comparison.png')
    plt.show()

def main():
    IMG_SIZE=100
    dataset = load_dataset()
    dataset = np.array(dataset)

    #Transform to a propper type
    img_original= dataset.astype('Float32')
    img_original =img_original/255.0
    plt.imshow(img_original[0])
    plt.show()
    #Give some gaussian noise to the imgs
    img_noisy = img_original + 0.1 * np.random.normal(loc=0.0, scale=1.0, size=img_original.shape)
    #Due to the value range we need to set limits, because it's possible to get values above 1.0
    img_noisy = np.clip(img_noisy, 0., 1.)
    plt.imshow(img_noisy[0])
    plt.show()

    #IMPORTANT
    #The NN take 1-D vector, so RGB images have to be transformed and take 3 different images, each
    #for chanel
    #Te quiero stackoverflow
    #https://stackoverflow.com/questions/42153826/what-is-meaning-of-transpose3-0-1-2-in-python
    #https://stackoverflow.com/questions/42153826/what-is-meaning-of-transpose3-0-1-2-in-python/62975651#62975651
    #THIS JUAN------___>https://stackoverflow.com/questions/40796985/what-does-the-line-of-code-np-transposeimage-tensor-2-1-0-do

    # (NUM_IMAGES,RES_X,RES_Y,RGB)
    #(NUM_IMAGES,100,100,3)---->(NUM_IMAGES,3,100,100)
    #transpose_clear=np.transpose(img_original,(0,3,1,2))
    #transpose_noise= np.transpose(img_noisy, (0, 3, 1, 2))

    #Now if we flat everything we will get 3 images more (kinda of)
    #Let's go to 1D
    #flat = transpose_clear.reshape(-1, IMG_SIZE*IMG_SIZE)
    #flat_N = transpose_noise.reshape(-1, IMG_SIZE * IMG_SIZE)

    # This is for saving the cnn with a different name eacht time is trained
    name = "denoissing_catsdogs_clas_cnn_64x2-{}".format(int(time.time()))

    tb = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(name))

    # Autoencoding
    #model = keras.Sequential()

    #model=create_model()
    model =create_model()

    model.summary()

    history=model.fit(img_noisy,img_original, epochs=10, batch_size=32, shuffle=True, callbacks=[tb], validation_data=(img_noisy,img_original),
)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


    #pickle_save = open("history.pickle", "wb")
    #pickle.dump(history, pickle_save)
    #pickle_save.close()

    model.save('NNden_model_newmodel.model')

    decoded_imgs = model.predict(img_noisy)

    show_images(5,img_noisy, decoded_imgs)

#if __name__ == "__main__":
#  main()

def deb():
    IMG_SIZE = 100
    dataset = load_dataset()
    dataset = np.array(dataset)

    # Transform to a propper type
    img_original = dataset.astype('Float32')
    img_original = img_original / 255.0

    # Give some gaussian noise to the imgs
    img_noisy = img_original + 0.1 * np.random.normal(loc=0.0, scale=1.0, size=img_original.shape)
    # Due to the value range we need to set limits, because it's possible to get values above 1.0
    img_noisy = np.clip(img_noisy, 0., 1.)


    model = tf.keras.models.load_model('NNden_model_newmodel.model')

    decoded_imgs = model.predict(img_noisy)

    n = 10

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(img_noisy[i])
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def plots():
    IMG_SIZE = 100
    dataset = load_dataset()
    dataset = np.array(dataset)

    # Transform to a propper type
    img_original = dataset.astype('Float32')
    img_original = img_original / 255.0

    # Give some gaussian noise to the imgs
    img_noisy = img_original + 0.1 * np.random.normal(loc=0.0, scale=1.0, size=img_original.shape)
    # Due to the value range we need to set limits, because it's possible to get values above 1.0
    img_noisy = np.clip(img_noisy, 0., 1.)


    # This is for saving the cnn with a different name eacht time is trained
    name = "denoissing_catsdogs_clas_cnn_64x2-{}".format(int(time.time()))

    tb = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(name))

    # Autoencoding
    # model = keras.Sequential()

    # model=create_model()
    model =tf.keras.models.load_model('NNden_model_newmodel.model')

    decoded_imgs = model.predict(img_noisy)

    plt.imshow(img_original[0])
    plt.show()

    plt.imshow(img_noisy[0])
    plt.show()

    plt.imshow(decoded_imgs[0])
    plt.show()

plots()







