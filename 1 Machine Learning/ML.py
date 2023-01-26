# SKILLS: Regression Classification Clustering Scikit learn Scipy

# SUPERVISED LEARNING
# - Regression/Estimation: continous values
# - Classification
# UNSUPERVISED LEARNING (less controled, fewer methods)
# - Dimension reduction/Feature reduction: Principal Component Analysis
# - Density estimation
# - Market basket analysis
# - Clustering: structure of data/summary/anomaly detection
# OTHERS:
# - Sequence mining: Markov model, HMM
# - Recommendation systems
# - Association: co-ocurring events

# Artificial Intelligence > Machine Learning (statistical branch) > Deep Learning

sklearn.preprocessing.StandardScaler().fit(X).transform(X)
sklearn.model_selection.train_test_split(X, Y, test_size = 0.3)
model = sklearn.svm.SVC(gamma=0.001, C=100.) .fit(x, y) .predict(x_test)
print(sklearn.metrics.confusion_matrix(y_test, yhat, labels=[1,0]))
pickle.dumps(model)