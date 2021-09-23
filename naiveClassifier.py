import numpy as np
from sklearn import linear_model, datasets
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from getFeatureVectors import getTrainTestSets
import pickle


with open('X_train.txt') as f:
    X_train = pickle.load(f)
with open('y_train.txt') as f:
    y_train = pickle.load(f)
with open('X_test.txt') as f:
    X_test = pickle.load(f)
with open('y_test.txt') as f:
    y_test = pickle.load(f)

#print len(X_train)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier( n_estimators = 100)
classifier = classifier.fit(X_train, y_train)
results = classifier.predict(X_test)


count = 0

wrong_indices = []
for j in range(len(results)):
    if results[j] != y_test[j]:
        count += 1
        wrong_indices.append(j)

<<<<<<< HEAD
print "accuracy", float(count)/len(results)
=======
print "accuracy", 1 - float(count)/len(results)
pickle.dump(wrong_indices, open('wrong_indices.txt', 'wb'))
>>>>>>> 5fe617c3787bc7476fa69ed14be9b1ff50dcce36
