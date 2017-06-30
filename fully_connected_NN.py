import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import scipy.io
import numpy as np
from keras.callbacks import ModelCheckpoint




# (get the data from files)
# read train data into python from .mat file
mat = scipy.io.loadmat('classifier_data.mat')
x_train = mat['TrainData_1d']
y_train = keras.utils.to_categorical(mat['Train_Lables'])
# read validation data into python from .mat file
x_val = mat['ValData_1d']
y_val = keras.utils.to_categorical(mat['Val_Lables'])
# read test data into python from a .mat file
x_test = mat['TestData_1d']
y_test = keras.utils.to_categorical(mat['Test_Lables'])


# (construct the Fully connectd model)
# define a sequential model
model = Sequential()
# a fully connected layer of 200 neuron and a relu activation
model.add(Dense(100, activation='sigmoid', input_dim=2920))
# to prevent overfitting we add a dropout TO THE OUTPUT OF THE PREVIOUS LAYER ONLY which shuts down 50% of the connections during training selected at random each time
model.add(Dropout(0.5))
# a fully connected layer of 50 neuron and a relu activation
model.add(Dense(50, activation='sigmoid'))
# a dropout layer
model.add(Dropout(0.25))
# the last 10 neurons "softmax" layer for classification
model.add(Dense(10, activation='softmax'))


# (configure the learnig methode and hyper-parameters "optimizer")
# applying Gradient Descent methode "lr=learnig rate" 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# apply previous confg. to our model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                          

# saving weights of the model after every epoch in a given directory
filepath="./weightsFCNN/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)


# using tensorbard to visualize the training and view learning curves to know when to stop and choose which epoch as the best
# while the code is running run the following command in your terminal while pointing to the script directory
# command -> (python3 -m tensorflow.tensorboard --logdir=./)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./', histogram_freq=0, write_graph=True)


# (training of the model)
# passing the training data and validation data along with how many examples to evaluate at a time "batch_size" and to loop over data how many times "epochs"
# and shuffle the data help for a faster convergence and better accuracies
model.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=32, epochs=100, verbose=2, shuffle=True, callbacks=[checkpoint,tbCallBack])


# (evaluate the performance of the model on the test data) after loading the best weights into the model by using the following line
# (model.load_weights('my_best_weight.hdf5')
performance=model.evaluate(x_test, y_test, batch_size=32, verbose=0)
print("your model accuracy is: ",performance[1]*100,'%')

