import scipy.io
import numpy as np
from sklearn import svm


# (get the data from files)
# read train data into python from .mat file
mat = scipy.io.loadmat('classifier_data.mat')
x_train = mat['TrainData_1d']
y_train = mat['Train_Lables']
# read validation data into python from .mat file
x_val = mat['ValData_1d']
y_val = mat['Val_Lables']
# read test data into python from a .mat file
x_test = mat['TestData_1d']
y_test = mat['Test_Lables']


# train the classifier (rbf kernel) with a one Vs. one strategy you can also use
# a one Vs. the rest "ovr"
TrainData = np.concatenate((x_train,x_val))
TrainLables = np.concatenate((y_train,y_val))
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(TrainData[0:20000], np.ravel(TrainLables[0:20000]))
print('finished training')


# evaluate the model accuracy
out = clf.predict(x_test)
accuracy = (np.sum(1*(out==np.ravel(y_test)))/5980)*100 
print("your model accuracy is: ",accuracy,'%')



# General notes:
# 1)This is a cpu imp. of SVM which is extremely slow to use with the total data size
#   even on the server so you have three options "use smaller data set for training"
#   like demonstrated in the above example or "use a dimensionality reducer first"
#   or "construct a keras binary classifier using one layer of linear nodes and hinge
#   loss function" and then design 45 classifiers and link them together for a one Vs.
#   all SVM 

