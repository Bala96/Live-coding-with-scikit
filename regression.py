### Machine Learning & Deep Learning Community
# Author: Manish Shivanandhan(manishshivanandhan@gmail.com)
# Regression example with Scikit
###
from sklearn import linear_model	# importing linear_model from scikit(http://scikit-learn.org/stable/modules/linear_model.html)
clf = linear_model.Lasso(alpha = 0.1)	# Using lasso regression as the classifier with 0.1 learning rate
clf.fit([[0, 0], [1, 1],[2,2],[3,3],[4,4],[5,5]], [0, 2,4,6,8,10])	# training the classifier with the data
print(clf.predict([[2.5, 2.5]]))	# prediction (4.8 - 5.2) 95%+ accuracy