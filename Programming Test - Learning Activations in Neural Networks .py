#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import pandas as pd
from keras.layers.advanced_activations import LeakyReLU
import seaborn as sns
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from keras.optimizers import Adam

# Test Backprop on Seeds dataset
# load and prepare data
file_list = ['Bank-Note', 'Iris', 'Breast Cancer', 'MNIST']

for file in file_list:
    if file == 'Bank-Note':
        num_classes = 1
        df = pd.read_csv('BankNote_Authentication.csv')
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
        i_shape = 4
        filt = 1
        loss_func = 'categorical_crossentropy'

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
        
    elif file == 'Iris':
        df = sns.load_dataset("iris")
        num_classes = 1
        X = df.iloc[:,0:4].values
        y = df.iloc[:,4].values
        i_shape = 4
        filt = 3
        loss_func = 'categorical_crossentropy'

        encoder =  LabelEncoder()
        y1 = encoder.fit_transform(y)
        Y = pd.get_dummies(y1).values

        X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=0) 

    elif file == 'Breast Cancer':
        df1= pd.read_csv('X_data.csv')
        df2 = pd.read_csv('Y_data.csv')
#         num_classes = 1
        X = df1.iloc[:,:-1]
        y = df2.iloc[:,-1]
        i_shape = 29
        filt = 13
        loss_func = 'binary_crossentropy'
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
        

    elif file == 'MNIST':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
#         num_classes = 10
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        y_train = keras.utils.to_categorical(y_train,512)
        y_test = keras.utils.to_categorical(y_test, 512)
        i_shape = 784
        filt = 512
        loss_func = 'categorical_crossentropy'
    batch_size = 128    
    epochs = 20
    scores = {}
    for activation in [None, 'sigmoid', 'tanh', 'relu']:
        model = Sequential()
        model.add(Dense(filt, activation = activation, input_shape=(i_shape,)))
        model.add(Flatten())
#         model.add(Dense(num_classes, activation = 'softmax'))
        model.compile(loss=loss_func,
                      optimizer=RMSprop(lr=0.04),
                      metrics=['accuracy'])
        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(X_test, y_test))
    
        score = model.evaluate(X_test, y_test, verbose=100)
        scores[activation] = score[1]
    
        plt.plot(history.history['val_accuracy'])
    
        plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['None', 'sigmoid', 'tanh', 'relu'], loc='upper left')


    max_key = max(scores, key = scores.get)
    max_value = max(scores.values())

    print(f"Dataset : {file} Best Performing Áctivation function : {max_key} having {max_value*100:0.2f}% accuracy")


# In[ ]:




