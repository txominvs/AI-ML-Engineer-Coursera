#Until 38
# SKILLS: Regression Classification Clustering Scikit learn Scipy

# SUPERVISED LEARNING
# - Regression/Estimation: continous values (simple regression 1 feature, multiple more features)
#       >> Ordinal, poisson, fast forest quantile, linear (analytic formulas!),
#       >> Polynomial, Lasso, Stepwise, ridge, bayesian linear, Neural Network,
#       >> Decision forest, boosted decision tree, K-Nearest Neighbors
#       ++ K-fold cross-validation (multiple train/test split and average accuracy)
#       ++ Metrics: MAE, MSE, RMSE, Relative Absolute Error (prediction error/mean value predictor)
#                   Relative Squared Error(prediction error squared/mean value deviation squared)
#                   R² metric = 1 - Relative Squared Error, the higher the better
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

dataframe = pandas.read_csv(); dataframe.head(); dataframe.describe();
dataframe[["column1","columname2"]].hist(); plt.show();
mask = np.random.rand(len(dataframe)) < 0.8; train, test = dataframe[mask], dataframe[~mask]

regr = sklearn.linear_model.LinearRegression(); regr.fit(x,y); n,m = regr.intercept_, regr.coef_
regr.predict(x); sklearn.metrics.r2_score(x,y) # ranges from [-inf worse, +1 best]