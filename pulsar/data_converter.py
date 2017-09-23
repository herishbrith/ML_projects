''' This file contains code that is reponsible for converting
data in .csv file to .mat file, with labels X_train, y_train,
X_test and y_test.'''

from numpy import genfromtxt
import numpy as np
import scipy.io as sio

csvFile = genfromtxt("HTRU_2.csv", delimiter=",")
np.random.shuffle(csvFile)

# Get total number of examples
m = csvFile.shape[0]

# Create a division percentage array
division_perc = [0.8, 0.2]

# Define the range of training examples
training_range_limit = int(m*division_perc[0])

# Declare training set, starting from example 0 upto 80%
X_train = csvFile[:training_range_limit,:9]
y_train = csvFile[:training_range_limit,8]

#Declare test set
X_test = csvFile[training_range_limit:,:9]
y_test = csvFile[training_range_limit:,8]

print(X_train.shape[0] + X_test.shape[0] == m)

# Save .mat file with a name and all numpy variables
sio.savemat('HTRU_2.mat', {
	"X_train": X_train,
	"y_train": y_train,
	"X_test": X_test,
	"y_test": y_test
})
