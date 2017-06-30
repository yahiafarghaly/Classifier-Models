# importing the required dep. first
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import scipy.io

# this is the size of our encoded representations
encoding_dim = 100

# this is our input placeholder "input size"
input_img = Input(shape=(2920,))

# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(2920, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# set optimizer and loss function
autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')

# read matlab data into python
mat = scipy.io.loadmat('classifier_data.mat')
Data = mat['TrainData_1d']
print (Data.shape)

# train the auto_encoder
autoencoder.fit(Data, Data,
                epochs=1,
                batch_size=128,
                shuffle=True,
                verbose=2)
                
# encode the images to lower dimension 
encoded_imgs = encoder.predict(Data)

# save as a matlab file
scipy.io.savemat('resized_data_AE.mat',{'Data':encoded_imgs})

# General notes:
# 1)when using AE as a dimensionality reducer you should use valdiation
#   and training data to train the AE so first concatinate them together
#   and then pass them to the (autoencoder.fit()) line above.

# 2)after traing the new samples "test samples" are passed through the model
#   using (encoder.predict()) line above.

# 3)the above code is just a demonstration of how the AE code work and used
#   data is only the training data which is WRONG.

