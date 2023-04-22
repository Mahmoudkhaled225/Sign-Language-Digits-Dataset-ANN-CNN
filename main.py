"""
1-Mohamed Ibrahim Shawky 20198068
2-mahmoud Khaled Helmy 20188045
3-mohamed hamada 20188041
4- Ghada Mamdouh Ahmed 20198063
"""
import keras
from keras.utils import  to_categorical
from PIL import Image, ImageStat
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam
from keras.models import Sequential
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
import os
import sys
import cv2
from random import shuffle
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=sys.maxsize)

def create_training_data_gray(dataset_path,categories,img_size):
    training_data = []
    for category in categories:
        path = os.path.join(dataset_path, category)
        class_num = categories.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        for img in os.listdir(path):  # iterate over each image per dogs and cats
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (img_size, img_size))  # resize to normalize data size
            training_data.append([new_array, class_num])  # add this to our training_data
    return training_data


def ANN_model_1(training_data):
    tempX = []
    tempY = []
    for features, label in training_data:
        tempX.append(features)
        tempY.append(label)
    #preprocessing
    X = np.array(tempX)
    Y = np.array(tempY)
    Y = Y.astype("float32")

    X = np.expand_dims(X, -1)
    lastXX = X.astype("float32")
    lastXX = lastXX * (1 / 255)
    s1,s2,s3,s4 =lastXX.shape
    lastXX = lastXX.reshape((s1, s2*s3*s4))

    #splitting data 80 20
    X_train, X_test, y_train, y_test = train_test_split(lastXX, Y, test_size=0.20, shuffle=True)

    # convert class vectors to binary class matrices
    y_trains = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    #model
    model = Sequential()
    model.add(Dense(128, activation='relu'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    # Compile Model
    model.build((None, 4096))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    #cross validation k-fold
    #we split the data into 5 folds
    #we split inside 10 90
    for i in range(5):
        print("At k_fold {}:".format(i + 1))
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_trains, test_size=0.1, random_state=42)
        model.fit(X_train1, y_train1, epochs=20, validation_split=0.2, batch_size=64)
        model.evaluate(X_test1, y_test1)

    y_pred = model.predict(X_test)
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    print("recall score")
    print(metrics.recall_score(y_test, y_pred, pos_label='positive', average='micro')*100)
    print("precision score")
    print(metrics.precision_score(y_test, y_pred, pos_label='positive', average='micro')*100)
    print("fscore ")
    print(metrics.f1_score(y_test, y_pred, pos_label='positive', average='micro')*100)
    print("accuracy score")
    print(metrics.accuracy_score(y_test, y_pred)*100)


def ANN_model_2(training_data):
    # np.random.shuffle(arr)
    tempX = []
    tempY = []
    for features, label in training_data:
        tempX.append(features)
        tempY.append(label)

    X = np.array(tempX)
    Y = np.array(tempY)
    Y = Y.astype("float32")

    X = np.expand_dims(X, -1)
    lastXX = X.astype("float32")
    lastXX = lastXX * (1 / 255)
    s1,s2,s3,s4 =lastXX.shape
    lastXX = lastXX.reshape((s1, s2*s3*s4))


    X_train, X_test, y_train, y_test = train_test_split(lastXX, Y, test_size=0.20, shuffle=True)

    # convert class vectors to binary class matrices
    y_trains = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Dense(200, activation='relu'))
    model.add(Dense(165, activation='relu'))
    model.add(Dense(149, activation='relu'))
    model.add(Dense(122, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    # Compile Model
    model.build((None, 4096))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    #cross validation k-fold
    #we split the data into 5 folds
    #we split inside 20 80
    for i in range(5):
        print("At k_fold {}:".format(i + 1))
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_trains, test_size=0.2, random_state=42)
        model.fit(X_train1, y_train1, epochs=13, validation_split=0.1, batch_size=128)
        model.evaluate(X_test1, y_test1)


    y_pred = model.predict(X_test)
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred,axis=1)
    print("recall score")
    print(metrics.recall_score(y_test ,y_pred,pos_label='positive',average='micro')*100)
    print("precision score")
    print(metrics.precision_score(y_test ,y_pred,pos_label='positive',average='micro')*100)
    print("fscore ")
    print(metrics.f1_score(y_test,y_pred,pos_label='positive',average='micro')*100)
    print("accuracy score")
    print(metrics.accuracy_score(y_test, y_pred)*100)


def training_RGB_data(img_folder):
    image_array = []
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = Image.open(image_path)
            stat = ImageStat.Stat(Image.open(image_path))
            average = stat.mean
            image = image.resize((100,100))
            image = np.array(image)
            image = image.astype('float32')
            image = image - average
            image /= 255
            image_array.append([image, int(dir1)])
    return image_array


def SVM_model(PIL_img_data):
    print("SVM model :")
    tempX=[]
    tempY=[]
    for features, label in PIL_img_data:
        tempX.append(features)
        tempY.append(label)
    X = np.array(tempX)
    Y = np.array(tempY)
    #split data into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20,shuffle=True)

    nsamples, nx, ny,nz = X_train.shape
    d2_train_dataset = X_train.reshape((nsamples, nx * ny*nz))
    nsamples, nx,ny,nz = X_test.shape
    d2_test_dataset = X_test.reshape((nsamples, nx*ny*nz))

    svclassifier = SVC(kernel='poly', degree=8)
    svclassifier.fit(d2_train_dataset, y_train)
    y_pred = svclassifier.predict(d2_test_dataset)
    acc = accuracy_score(y_test, y_pred)*100
    print("SVM model accuracy is ",acc)
    return acc



def CNN_model(PIL_img_data) :
    print("CNN model :")
    tempX=[]
    tempY=[]
    for features, label in PIL_img_data:
        tempX.append(features)
        tempY.append(label)
    X = np.array(tempX)
    Y = np.array(tempY)
    #split data into Train and Test
    X_train, X_test, y_train, y_test =train_test_split(X,Y, test_size=0.20,shuffle=True)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)


    CNN_model = Sequential()
    CNN_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3), padding='same'))
    CNN_model.add(MaxPooling2D((2, 2), padding='same'))
    CNN_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    CNN_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    CNN_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    CNN_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    CNN_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    CNN_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    CNN_model.add(Flatten())
    CNN_model.add(Dense(128, activation='relu'))
    CNN_model.add(Dense(10, activation='softmax'))
    CNN_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    batch_size = 64
    epochs = 15
    for i in range(3):
        print("At k_fold {}:".format(i + 1))
        X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train,y_train , test_size=0.1, random_state=42)
        CNN_model.fit(X_train1, y_train1, epochs=5, validation_split=0.2, batch_size=64)
        CNN_model.evaluate(X_test1, y_test1)


    y_pred = CNN_model.predict(X_test)
    y_pred = np.argmax(y_pred,axis=1)
    y_test = np.argmax(y_test, axis=1)
    print("recall score")
    print(metrics.recall_score(y_test ,y_pred,pos_label='positive',average='micro')*100)
    print("precision score")
    print(metrics.precision_score(y_test ,y_pred,pos_label='positive',average='micro')*100)
    print("fscore ")
    print(metrics.f1_score(y_test,y_pred,pos_label='positive',average='micro')*100)
    print("accuracy score")
    acc = metrics.accuracy_score(y_test, y_pred)*100
    print(acc)
    return acc



def main():
    dataset_path = "C:\\Users\\Abo El-Magd\\Desktop\\assignment 2 ML\\Sign-Language-Digits-Dataset-master\\Dataset"
    categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    img_size = 64
    training_gray_data = create_training_data_gray(dataset_path, categories, img_size)
    shuffle(training_gray_data)
    print("ANN model 1: ")
    ANN_model_1(training_gray_data)
    print("ANN model 2: ")
    ANN_model_2(training_gray_data)

    '''
    CNN model and SVM model (run on RGB images)
    '''
    rgb_img_data = training_RGB_data(dataset_path)
    shuffle(rgb_img_data)
    CNN_acc = CNN_model(rgb_img_data)
    SVM_acc = SVM_model(rgb_img_data)

    if CNN_acc > SVM_acc:
        print("As we see :")
        print("CNN model accuracy is ", CNN_acc)
        print("and SVM model accuracy is ", SVM_acc)
        print(
            "so CNN model is batter than SVM and complex models (CNN) can deal with complex data and get great accuracy but simple models (SVM) can't")


if __name__ == "__main__":
    main()

