"""
Created on Tue Aug 20 12:05:06 2024
@author: wilbertoperezzamora

This script implements the kNN (k-Nearest Neighbor) algorithm for digit
classification.
"""

import pickle
from tkinter.filedialog import askopenfilename, asksaveasfilename
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# READING DATA.
with open(askopenfilename(title='Select the image properties file.'),"rb") as f:
    train_labels,train_img_flatted,train_properties = pickle.load(f)
with open(askopenfilename(title='Select the image properties file.'),"rb") as f:
    test_labels,test_img_flatted,test_properties = pickle.load(f)

# k-NEAREST NEIGHBOR ALGORITHM.
kNN = KNeighborsClassifier(n_neighbors = 3)

# TRAINING.
kNN.fit(train_properties,train_labels)

# TESTING.
predictions = kNN.predict(test_properties)

# METRICS.
CMtx = confusion_matrix(test_labels,predictions) # Confusion matrix.
accuracy = accuracy_score(test_labels,predictions) # Model accuracy.

sensibility = []
specificity = []

# TP, FN, FP and TN FOR EVERY CLASS (DIGITS FROM ZERO TO NINE)
for index in range(len(CMtx)):
    TP = CMtx[index,index]  # True positives.
    FN = sum(CMtx[index,:]) - TP  # False negatives.
    FP = sum(CMtx[:,index]) - TP  # False positives.
    TN = CMtx.sum() - (TP + FN + FP)  # True negatives.

    # SENSIBILITY.
    class_sensibility = TP / (TP+FN) if (TP+FN) != 0 else 0
    sensibility.append(class_sensibility)
    # SPECIFICITY.
    class_specificity = TN / (TN+FP) if (TN+FP) != 0 else 0
    specificity.append(class_specificity)
    
# SAVING DATA.
with open(asksaveasfilename(title='Select the path to save the results.'),"wb") as f:
    pickle.dump(CMtx,f)
    pickle.dump(accuracy,f)
    pickle.dump(sensibility,f)
    pickle.dump(specificity,f)