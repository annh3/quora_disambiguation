import numpy as np
import pickle
import os
from getFeatureVectors import getTrainTestSets

X_train, y_train, X_test, y_test = getTrainTestSets()

pickle.dump(X_train, open('X_train.txt', 'wb'))
pickle.dump(y_train, open('y_train.txt', 'wb'))
pickle.dump(X_test, open('X_test.txt', 'wb'))
pickle.dump(y_test, open('y_test.txt', 'wb'))