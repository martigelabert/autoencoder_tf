import tensorflow as tf
from tensorflow import keras
import cv2
import os
import numpy as np
import random
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
import matplotlib.pyplot as plt
from skimage import color
import pickle
import time
from tensorflow.keras.models import Sequential
import  varname
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


def add_noise(x_train, x_test, noise_factor):
    #noise_factor = 0.4
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return (x_train_noisy, x_test_noisy )

def simple_load():
    IMG_SIZE = 100
    training_imgs = []
    test_imgs = []
    original_imgs = []
    directory_train = r"/home/marti/Desktop/Datasets/birds/525159_963647_bundle_archive/birds_train"
    classification = os.listdir(directory_train)

    for category in classification:
        path = os.path.join(directory_train, category)
        class_num = classification.index(category)
        for img in os.listdir(path):
            try:

                array_img = cv2.imread(os.path.join(path, img))

                original_imgs.append(array_img)

                new_img_array = cv2.resize(array_img, (IMG_SIZE, IMG_SIZE))

                training_imgs.append(new_img_array)
                test_imgs.append([new_img_array, class_num])
            except Exception as error:
                pass
        print("Load of ", category, " completed")


    return training_imgs

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

def save_for_fist_time():
    IMG_SIZE = 100
    print("This is a demo")
    training_imgs= simple_load()
    random.shuffle(training_imgs)

    img_test = []
    img_train = []

    img_train_n=[]
    img_test_n=[]

    n_level = [0.1,0.2,0.3,0.4]

    for img in training_imgs:
        img_train.append(img)
        img_test.append(img)

    # -1 is just for saying how many elements we have? in this case we don't know
    # IMG_SIZE is the dimension of X and Y axis of the image(data)
    # 1 is for sayiing that we are in gray scale and is just 1 value, if we
    # want to do it with color the value would be 3 (RGB)
    img_train = np.array(img_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    img_test = np.array(img_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


    #Estoy infinitamente seguro que puedo optimizar esto, pero no creo que lo vaya a hacer
    for n in range(len(n_level)):
        img_train_n[n],img_test_n[n] =add_noise(img_train, img_test, n_level[n])

    for n in range(len(n_level)):
        img_train_n[n] = np.array(img_train_n[n]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        img_test_n[n] = np.array(img_test_n[n]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


    #QUEDA GUARDAR TODAS LAS IMAGENES CON RUIDO
    save_data(img_train, img_test,varname.varname(img_train), varname.varname(img_test))


def load_saved_data():
    pk_opend = open("img_x.pickle", "rb")
    x = pickle.load(pk_opend)

    pk_opend = open("label_y.pickle", "rb")
    y = pickle.load(pk_opend)

    return x, y



def first_train():

    # Set CPU as available physical device
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

    # To find out which devices your operations and tensors are assigned to
    #tf.debugging.set_log_device_placement(True)

    # This is for saving the cnn with a different name eacht time is trained
    name = "Bird_clas_cnn_64x2-{}".format(int(time.time()))

    tb = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(name))

    img_x, label_y = load_saved_data()
    print("Se han cargado los datos")

    img_x = img_x / 255.0
    label_y=np.array(label_y)
    # print(img_x[1])

    #I don't need the 8000 value from the array so, we will omit them with
    print(img_x.shape[1:])
    # make model
    model = Sequential()

    model.add(keras.layers.Conv2D(128, (3, 3), activation= "relu", input_shape=img_x.shape[1:]))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(64, (3, 3),activation= "relu"))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(64, (3, 3),activation= "relu"))
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    # We need flatter our data because conv2D is in fact
    # in 2D
    model.add(keras.layers.Flatten())
    # Dense layer needs one-D shape
    model.add(keras.layers.Dense(32,activation= "relu"))
    model.add(keras.layers.Dense(64, activation= "sigmoid"))

    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics =['accuracy'])
    model.fit(img_x, label_y, epochs=12, batch_size=64,shuffle=True, callbacks=[tb])

    pk_opend = open("table_int.pickle", "rb")
    table_int= pickle.load(pk_opend)

    pk_opend = open("table_string.pickle", "rb")
    table_string= pickle.load(pk_opend)

    pk_opend = open("original_imgs.pickle", "rb")
    original_img = pickle.load(pk_opend)


    p=model.predict(img_x)
    print(p)

    #The argmax just gets the value that has more probability
    print(table_string[table_int.index(np.argmax(p[33]))])

    plt.imshow(img_x[33])
    plt.show()

    plt.imshow(original_img[33])
    plt.show()

    # Drawing images for comparison
    n = 4
    plt.figure(figsize=(70, 70))
    for i in range(n):
        plt.title(table_string[table_int.index(np.argmax(p[i]))])
        plt.subplot(2, n, i + 1)
        plt.imshow(img_x[i], cmap="binary")

        plt.title(table_string[i])
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(original_img[i], cmap="binary")

    plt.show()

    model.save('bird_clas_model_newmodel.model')
    # If it's your first time, save first
    #save_for_fist_time()


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

#showing_prediction()
first_train()

