import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np
import random

import matplotlib.pyplot as plt
from skimage import color
import pickle
import time
from tensorflow.keras.models import Sequential
import  varname
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, UpSampling2D

#Use noise_factor 0.0 < x < 1.0
def add_noise(x_train, x_test, noise_factor):
    #noise_factor = 0.4
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return (x_train_noisy, x_test_noisy )
#Retuns an array of images
def simple_load():
    IMG_SIZE = 100
    dataset = []
    directory = r"/home/marti/Desktop/Datasets/Catsvsdogs/test"
    for img in os.listdir(directory):
        try:
            array_img = cv2.imread(os.path.join(directory, img))
            new_img_array = cv2.resize(array_img, (IMG_SIZE, IMG_SIZE))
            dataset.append(new_img_array)
        except Exception as error:
            pass
    print("Load of ", img, " completed")
    return dataset
#Save test and train images (NO NOISE)
def save_data(img_train, img_test, var1_name, var2_name):

    file1=var1_name+".pickle"
    file2=var2_name+".pickle"
    ######SAVING DATA#######
    pickle_save = open(file1, "wb")
    pickle.dump(img_train, pickle_save)
    pickle_save.close()

    pickle_save = open(file2, "wb")
    pickle.dump(img_test, pickle_save)
    pickle_save.close()
    ###END SAVING DATA###
    return 0
#Save test and train images (NO NOISE)
def save_dataset(dataset, var1_name):
    file1=var1_name+".pickle"
    ######SAVING DATA#######
    pickle_save = open(file1, "wb")
    pickle.dump(dataset, pickle_save)
    pickle_save.close()
    ###END SAVING DATA###
    return 0
#FIRST FUNCTION
def save_for_fist_time():
    IMG_SIZE = 100
    print("This is a demo")
    dataset= simple_load()

    #Shuffle data for future training
    random.shuffle(dataset)

    img_test = []
    img_train = []

    img_train_n=[]
    img_test_n=[]

    n_level = [0.1,0.2,0.3,0.4]

    #This have O(n) but Idont care, giving random noise  to the images
    for img in dataset:
        noise = random.randrange(len(n_level))
        x_train_noisy = img + noise * np.random.normal(loc=0.0, scale=1.0, size=img.shape)
        x_test_noisy = img + noise * np.random.normal(loc=0.0, scale=1.0, size=img.shape)

        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
        x_test_noisy = np.clip(x_test_noisy, 0., 1.)

        img_train.append(x_train_noisy)
        img_test.append(x_test_noisy)


    # -1 is just for saying how many elements we have? in this case we don't know
    # IMG_SIZE is the dimension of X and Y axis of the image(data)
    # 1 is for sayiing that we are in gray scale and is just 1 value, if we
    # want to do it with color the value would be 3 (RGB)
    img_train = np.array(img_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    img_test = np.array(img_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


    #QUEDA GUARDAR TODAS LAS IMAGENES CON RUIDO
    save_data(img_train, img_test,varname.varname(img_train), varname.varname(img_test))
    save_dataset(dataset,varname.varname(dataset))
def load_saved_data():
    pk_opend = open("img_train.pickle", "rb")
    x = pickle.load(pk_opend)

    pk_opend = open("img_test.pickle", "rb")
    y = pickle.load(pk_opend)

    pk_opend = open("dataset.pickle", "rb")
    z = pickle.load(pk_opend)

    #train, test, original
    return x, y, z
def first_train():

    # Set CPU as available physical device
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

    # To find out which devices your operations and tensors are assigned to
    #tf.debugging.set_log_device_placement(True)

    # This is for saving the cnn with a different name eacht time is trained
    name = "denoissing_catsdogs_clas_cnn_64x2-{}".format(int(time.time()))

    tb = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(name))

    img_x, img_y, img_z = load_saved_data()
    print("Se han cargado los datos")

    # train, test, original
    img_x = img_x / 255.0
    img_y = img_y / 255.0
    #img_z have the clean images
    img_z = img_z / 255.0

    # print(img_x[1])

    #I don't need the 8000 value from the array so, we will omit them with
    print(img_x.shape[1:])

    # make model
    # Autoencoding
    model = Sequential()
    # Codification
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

    model.fit(img_x, img_z, epochs=12, batch_size=64,shuffle=True, callbacks=[tb])

    p=model.predict(img_y)
    print(p)

    plt.show()
    model.save('den_custom_100x100.model')
def showing_prediction():

    img_x, label_y = load_saved_data()
    print("Se han cargado los datos")

    img_x = img_x / 255.0
    label_y = np.array(label_y)
    model=tf.keras.models.load_model('bird_clas_model_newmodel.model')

    pk_opend = open("table_int.pickle", "rb")
    table_int= pickle.load(pk_opend)

    pk_opend = open("table_string.pickle", "rb")
    table_string= pickle.load(pk_opend)

    pk_opend = open("original_imgs.pickle", "rb")
    original_img = pickle.load(pk_opend)


    p=model.predict(img_x)
    print(p)



    # Drawing images for comparison
    n = 4
    plt.figure(figsize=(20, 20))
    for i in range(n):

        plt.title(table_string[table_int.index(np.argmax(p[i]))])
        plt.subplot(2, n, i + 1)
        plt.imshow(img_x[i], cmap="binary")
        print(table_string[i])
        plt.title(table_string[i])
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(original_img[i], cmap="binary")

    plt.show()


def main():
    save_for_fist_time()

if __name__ == "__main__":
    main()




