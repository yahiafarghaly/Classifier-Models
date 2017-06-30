import scipy.io
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# (get the data from files)
# read train data into python from .mat file
mat = scipy.io.loadmat('classifier_data.mat')
x_train = mat['TrainData_2d']
y_train = keras.utils.to_categorical(mat['Train_Lables'],num_classes=10)
# read validation data into python from .mat file
x_val = mat['ValData_2d']
y_val = keras.utils.to_categorical(mat['Val_Lables'],num_classes=10)
# read test data into python from a .mat file
x_test = mat['TestData_2d']
y_test = keras.utils.to_categorical(mat['Test_Lables'],num_classes=10)


# (construct the CNN model)
# define a sequential model
model = Sequential()
# adding a filters layer with 32 filter of size (3*3) and a relu activation function for each neuron
# and since this is the first layer we have to define the input shape in our case the input is a (20*146) picture with 1 channel
model.add(Conv2D(32, (3, 3), activation='relu',data_format='channels_first' ,input_shape=(1,20,146)))
# adding another filters layer with 32 filters of size (3*3) and a relu activation function for each neuron
model.add(Conv2D(32, (3, 3), activation='relu'))
# this is a sampling layer of a window size (2*2) which means out of each 4 neurons in the filters before we select one
model.add(MaxPooling2D(pool_size=(2, 2)))
# to prevent overfitting we add a dropout TO THE OUTPUT OF THE PREVIOUS LAYER ONLY which shuts down 25% of the connections during training selected at random each time
model.add(Dropout(0.25))
# another filters layer
model.add(Conv2D(64, (3, 3), activation='relu'))
# another filter layer
model.add(Conv2D(64, (3, 3), activation='relu'))
# another sampling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# a dropout layer
model.add(Dropout(0.25))
# flat the previous output into a single vector "prepare for a fuuly connected part of netwrok"
model.add(Flatten())
# a fully connected layer of 256 neuron and a relu activation
model.add(Dense(256, activation='relu'))
# a dropout for previous layer 
model.add(Dropout(0.5))
# the last 10 neurons "softmax" layer for classification
model.add(Dense(10, activation='softmax'))


# (configure the learnig methode and hyper-parameters "optimizer")
# applying Gradient Descent methode "lr=learnig rate" 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# apply previous confg. to our model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# saving weights of the model after every epoch in a given directory
filepath="./weightsCNN/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)


# using tensorbard to visualize the training and view learning curves to know when to stop and choose which epoch as the best
# while the code is running run the following command in your terminal while pointing to the script directory
# command -> (python3 -m tensorflow.tensorboard --logdir=./)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./', histogram_freq=0, write_graph=True)


# (training of the model)
# passing the training data and validation data along with how many examples to evaluate at a time "batch_size" and to loop over data how many times "epochs"
# and shuffle the data help for a faster convergence and better accuracies
model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=32, epochs=3, verbose=2, shuffle=True, callbacks=[checkpoint,tbCallBack])


# (evaluate the performance of the model on the test data) after loading the best weights into the model by using the following line
# (model.load_weights('my_best_weight.hdf5')
performance=model.evaluate(x_test, y_test, batch_size=32, verbose=0)
print("your model accuracy is: ",performance[1]*100,'%')


# General notes
# 1)this is only a code example the acc. is evaluated on the last epoch weights but you should not do that instead choose the weights
#   file with the highest val_acc by your eyes from the folder at which they are saved or use the tensorboard and load them into the model
#   using the above weights and then do (model.evaluate)

# 2)the above parameters are not optimum you should try to change them and add or remove layers to the design

# 3)change any thing but the loss function :D because this is the standard best one for classification problems


