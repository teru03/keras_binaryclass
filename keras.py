from tensorflow.contrib.keras.python.keras.models import Sequential,load_model
from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.layers import Dropout, Activation
from tensorflow.contrib.keras.python.keras.optimizers import Adagrad,SGD
from tensorflow.contrib.keras.python.keras.initializers import random_uniform

from fbas import FeatureBinarizatorAndScaler

import os
import pandas as pd
import numpy as np
import tensorflow as tf

X_train = pd.read_csv('./input/train.csv')
y_train = X_train['target']
X_test = pd.read_csv('./input/test.csv')
outdf = X_test['id']
X_test = X_test.drop(['id'], axis=1)
X_train = X_train.drop(['id', 'target'], axis = 1)
y_train1 = abs(-1+y_train)
y_train = pd.concat([y_train, y_train1], axis=1)
binarizerandscaler = FeatureBinarizatorAndScaler()
binarizerandscaler.fit(X_train)
X_train = binarizerandscaler.transform(X_train)
X_test = binarizerandscaler.transform(X_test)
y_train = y_train.as_matrix()


#hyperparameters
input_dimension = 226
learning_rate = 0.0025
momentum = 0.85
#hidden_initializer = random_uniform(seed=SEED)
tf.set_random_seed(32)
hidden_initializer = tf.random_uniform([1])
dropout_rate = 0.2

savefile = './models.h5'


# create model
if os.path.isfile(savefile):
     model = load_model(savefile)
else:
    model = Sequential()
    model.add(Dense(128, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu'))
    #model.add(Dense(128, input_dim=input_dimension,  activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, kernel_initializer=hidden_initializer, activation='relu'))
    model.add(Dense(2, kernel_initializer=hidden_initializer, activation='softmax'))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dense(2, activation='softmax'))

    sgd = SGD(lr=learning_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])

model.fit(X_train, y_train, epochs=50, batch_size=128)
predictions = model.predict_proba(X_test)

ans = pd.DataFrame(predictions,columns=['target','dmy'])
print ans
outdf = pd.concat([outdf,ans['target']],axis=1)
outdf.to_csv('./submit_keras.csv',index=False)


    #save the model
#model_json = model.to_json()
#with open("./ans.json", "w") as json_file:
#    json_file.write(model_json)

model.save(savefile)

