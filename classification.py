### Machine Learning & Deep Learning Community
# Author: Manish Shivanandhan(manishshivanandhan@gmail.com)
# Classification example with Scikit
###
import numpy as np 	#numpy module(http://www.numpy.org/)
import matplotlib.pyplot as plt 	#matplotlib for plotting graphs(http://matplotlib.org/)
from sklearn import svm #importing support vector machine from scikit(http://scikit-learn.org/stable/modules/svm.html)

# initial dataset x & y without labels
x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

# uncomment this if you want to see the scatter plot
# plt.scatter(x,y)
# plt.show()

# creating a numpy array X with [x,y] as input and Y as labels. 
# labels 0 & 1 represent the position of the points x,y in the plot. Top is 1 , bottom is 0
X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])
y = [0,1,0,1,0,1]

# support vector classifier with linear kernel.
clf = svm.SVC(kernel='linear', C = 1.0)
# training the classifier with the data
clf.fit(X,y)
# prediction (98%+ accuracy)
print(clf.predict([7,9]))