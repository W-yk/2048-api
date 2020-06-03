import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np

import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model , load_model

from keras.callbacks import ModelCheckpoint


board = Input(shape=(16,1))
l1 = LSTM(256)(board)  # the output will be a vector
out = Dense(4, activation='softmax')(l1)
agent = Model(inputs=board, outputs=out)
#agent.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[keras.metrics.categorical_accuracy])  
agent = load_model('model.hdf5')
a = np.loadtxt(open('./DATA.csv', "r"), dtype=int, delimiter=",")
X = np.array(a[:,0:16]).reshape(-1,16,1)
Y = np.eye(4)[a[:,-1]]


checkpoint = ModelCheckpoint(filepath='./logs/model-{epoch:02d}.hdf5', period=1)

agent.fit(X,Y,epochs =5,callbacks=[checkpoint])
agent.save('my_model.h5')
