# SKILLS: Regression Classification Clustering Scikit learn sklearn Scipy

# SUPERVISED LEARNING
# - Regression/Estimation: continous values (simple regression -> 1 feature ; multiple regression -> more features)
#       >> Ordinal, poisson, fast forest quantile, linear (analytic formulas! or Ordinary Least Squares or Gradient Descent),
#       >> Polynomial, Lasso, Stepwise, ridge, bayesian linear, Neural Network,
#       >> Decision forest, boosted decision tree, K-Nearest Neighbors
#       >> Regression tree (see Decision Trees)
#           Make branching each time
#           The prediction of each branching is the AVERAGE VALUE of the datapoints
#           Branchings are chosen so that the Mean Absolute Error SUM |real-predic| is minimized,
#           so the branching option with the least MAE is picked, among all options for:
#              Categorical Feature branching
#              Numerical Feature branching: binary branching with FEATURE < BOUNDARY and FEATURE >= BOUNDARY. All boundaries are tested between the i -th and i+1 -th datapoints, and the boundary with the LEAST MAE is chosen
#           Stoping conditions: 1) tree depth 2) remaining samples on a branch 3) number of samples if another branching is made
#       ++ K-fold cross-validation (multiple train/test split and average accuracy)
#       ++ Metrics: MAE, MSE, RMSE, Relative Absolute Error (prediction error/mean value predictor)
#                   Relative Squared Error(prediction error squared/mean value deviation squared)
#                   R^2 metric = 1 - Relative Squared Error, the higher the better
# - Classification: categorical variable "class"
#       >> Decission Trees
#           Nodes = branching of training dataset based on a category (Recursive Partitioning)
#           Branching of the data must group together as much as possible (Purity of the Leaves) datapoints with the same "prediction label"
#           Entropy of a group of data = - SUM(label) P(label) * log P(label) where P(label)=how many points belong to label/all datapoints
#           Entropy [0 data is very pure, +1 data is very random] = amount of admixture in a group
#           Each branching increases INFORMATION GAIN as much as possible = entropy before splitting - 1/number of datapoints * SUM(split) size of chunk * entropy of each chunk
#       >> Naïve Bayes, Linear Discriminant Analysis
#       >> K-Nearest Neighbor (KNN)
#           Select the K datapoints closest to the one we want to predict
#           Prediction = most popular class among its K neighbors
#           How to choose K? Use the TEST set to make a plot with ACCURACY vs. K-VALUE an choose the highest
#           For REGRESSION prediction = mean value of its K closest datapoints
#       >> Logistic Regression
#           Continous features (or one-hot encoded) -> BINARY categorical classification and PROBABILITY
#           Linear boundaries for classifing data
#           Makes a linear regression THROUGH the binary categorical data and then applies a SIGMOID 1/(1+exp(-x))
#           Cost function (loss function + regularization) = label - Probability(label)
#           Uses GRADIENT DESCENT is used to minimize loss and STOPPING CONDITION=accuracy: new params = old params - step size * d[cost function]/d[params]
#           Loss function for SIGMOID = LOG LOSS = -log[ Probability(correct label | input) ] and not the usual MEAN SQUARED ERROR SUM (y-pred)^2
#       >> Neural Networks
#       >> Support Vector Machines (SVM)
#           First, map data to a HIGHER DIMENSIONAL SPACE using KERNELLING: linear, polynomial, Radial Basis Functions, Sigmoid
#           Then, separate data using a HYPERPLANE (once the kernel turns it into LINEARLY SEPARABLE DATA)
#           The hyperplane is chosen so that the classes are MAXIMALLY SEPARATED
#           SUPPORT VECTORS: datapoints closes to the separation hyperplane, only these are used in the learning process
#               y = (w^T)*x + b where y >=1 represents one class and y<=-1 represents the other, equalities are for closest support vectors
#           AVANTAGES: well in high dimensions, only use a subset of the data (support vectors)
#           DISADVANTAGES: over fitter if features>samples, no class probabilities, computationally costly for big DB
#           USES: Image analysis, text filtering, genetics because many features
#       >> Softmax regression (multinomial logistic regression)
#           MULTICLASS classification: more possible labels than a binary classification, but each sample is associated with a single class
#           label_probabilities = [ exp(- A* X) , exp(- B* X) , exp(- C* X) ] / normalization
#           predicted_label = np.argmax(label_probabilities)
#           TRAINING according to logistic regression CROSS ENTROPY -> needs probabilties -> does *not* work with Support Vector Machines
#           Space is divided into HYPERPLANES A*X, B*X, C*X and regions correspond to labels
#       >> One vs All (One vs Rest): turn two-class binary classifier into MULTICLASS classifier
#           Let's say we have N classes. For each of the classes:
#               We train a binary classifier: CURRENT CLASS vs ALL THE OTHER CLASSES
#           Finally, we choose the class by either MAJORITY VOTE (the class that got most votes) or CLASSIFIER WITH THE HIGHEST PROBABILITY
#           AMBIGUITY CAN OCUR: resolve by selecting the CLASSIFIER that got the HIGHEST PROBABILITY (argmax) or use the FUSION RULE (add output of the linear classifier)
#       >> One vs One
#           For i = class label:
#               For j = class label > i:
#                   BinaryClassifier = does the sample belong to I or to J ?
#           Finally, MAJORITY VOTE = the class that got selected more times wins
#           In total, if we have N classes, we need to train aprox N^2 classifiers because SUM(i<=N) N-i = i(i-1)/2!. Or pick i=any of the N classes, j=any of the remaining N-1 classes, each pair will be repeated 2! times
#           SMALLER AMBIGUITY REGION: resolve by PROBABILITY or FUSION RULE (see one vs all classifiers)
#       ++ Metrics: Jaccard index [0 worst, +1 best]
#                                   = size of INTERSECTION prediction and label / size of UNION prediction and label
#                                   = correct predictions / (size of prediction + size of labels - size of correct predictions)
#       ++ Metrics: F1 score [0 worst, +1 best]
#            Step 1) Make confussion matrix
#              | predicted = A   |   predicted = B
#              ------------------------------------
# actual = A   |      TP                 FN
# actual = B   |      FP                 TN
#
#             TRICK! Each ROW of Confussion Matrix shows the ability to classify an individual class
#             Precission = accuracy to classify actual A             = TP/(TP+FN)
#             Recall     = how much can we trust a prediction of A   = TP/(TP+FP)
#             F1 score   = harmonic average of precission and recall = 0.5*1/(1/Precission + 1/Recall)
#       ++ Metrics: Log loss [0 best, +1 worst]
#             LogLoss    = average for all dataset of (-1) * is label active * log[ predicted_probability(label) ]

# UNSUPERVISED LEARNING (less controled, fewer methods)
# - Dimension reduction/Feature reduction: Principal Component Analysis
# - Density estimation
# - Market basket analysis
# - CLUSTERING: structure of data/summary (data compression)/anomaly detection (noise removal)
#       Unlabeled data SEGMENTATION (groups with similar features): recommend similar products, anomalies in bank transaction, genetic groups
#       >> Partition based clustering:
#           Efficient for large DB and sphere-like clusters
#           ++ k-Median, fuzzy c-Means
#           ++ k-MEANS
#               K = number of clusters (parameter)
#               1) Randomly choose K initial CLUSTER CENTERs: either some random poisition or some random datapoints
#               2) Assign the closest CLUSTER CENTER to each DATAPOINT using either Euclidean distance = SQRT (point-center)^2 or Minkowsky distance or cosine similarity. Creating a "distances matrix" is useful for this purpose
#               3) Update positions of CLUSTER CENTERs: new position = MEAN[ position of datapoints that are closest to this center ]
#               4) This approach will iteratively minimize Sum of Squared Error = SUM (point-center)^2
#               5) Go back to 1) until they CLUSTER CENTERS do not move significantly.
#               Heuristic algorithm -> may converge to LOCAL minima -> make several runs with different initial conditions
#               Metrics: smaller is better
#                   Average distance between DATAPOINTS WITHIN THE SAME CLUSTER
#                   Average distance between DATAPOINT AND THE CENTROID OF ITS CLUSTER
#               Optimal K: ELBOW METHOD
#                   For large K, average distance decreases as 1/SQRT(number of clusters)
#                   Choose the K value where the average distance goes from SHARPLY decreasing to MILDLY decreasing: ELBOW POINT
#       >> Hierarchical clustering
#           Intuitive tree classification, good for small DB
#           ++ Agglomerative algorithms, Divisive algorithms
#       >> Density-based clustering
#           Spatial clusters & noise, arbitrary shaped clustering
#           ++ DBSCAN
# OTHERS:
# - Sequence mining: Markov model, HMM
# - Recommendation systems
# - Association: co-ocurring events

# Artificial Intelligence > Machine Learning (statistical branch) > Deep Learning

sklearn.preprocessing.StandardScaler().fit(X).transform(X) # or StandardScaler().fit_transform(X)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.3)
sklearn.preprocessing.LabelEncoder.fit(['group 1', 'group 2', 'group 3']).transform(X)
sklearn.preprocessing.normalize(X, axis=1, norm='l1', copy=False) #axis=1 means that each ROW will be normalized as a vector
sklearn.datasets.load_iris().data .target; # data format [each sample [features], [next sample's features] ...]
coordinates, which_cluster_they_belong = sklearn.datasets.make_blobs(n_samples="how many points", centers=["center positions [1,2] [-1,2]"], cluster_std="size of each cluster")

dataframe = pandas.read_csv("file_name.csv", delimiter=","); dataframe.shape; dataframe.describe(); dataframe.dtypes; dataframe["categorical_column_name"].value_counts();
dataframe[dataframe['color'] == 'red'][0:5]; dataframe.name_of_the_column; dataframe[dataframe.columns].values[0:5]; dataframe.head();
dataframe[["column1","columname2"]].hist(); plt.show(); dataframe[["column_name"]].unique();
plt.hist(dataframe[["column1"]].values, 6, histtype='bar', facecolor='g'); plt.show()
plt.pie(dataframe[["column_name"]].value_counts().values, labels=dataframe[["column_name"]].unique(), autopct='%1.3f%%'); plt.show()
dataframe.iloc[row_number_slice, column_number_slice]
dataframe.isna().sum(); data.dropna(inplace=True) # Detect missing values: isna() replaces each value with True/False and sum() sums all ROWs into a single one. Dropna = remove rows with missing data
dataframe.groupby('name of the column').mean()
dataframe.drop(columns=["column_names"]); dataframe.drop(['column_names'], axis=1)
dataframe.abs().mean() # Useful for calculating MAE: abs() replaces values with abs and mean into a single ROW
dataframe[(dataframe['column_name'] >= 30) & (dataframe['another_column'] < 200)]
dataframe['time_column'] = pandas.to_datetime(dataframe['time_column']); dataframe['time_column'].dt.hour # also dt.weekday and some_date-anther_date_substract
dataframe = dataframe[pd.to_numeric(dataframe['column_name'], errors='coerce').notnull()]; dataframe['column_name'] = dataframe['column_name'].astype('int') # drop non-numerical valued rows and convert the rest to ints
pd.get_dummies(dataframe, columns = ['categorical_column_name']) # creates many columns to capture the ONE-HOT ENCODING


mask = np.random.rand(len(dataframe)) < 0.8; train, test = dataframe[mask], dataframe[~mask]
x = np.asanyarray(train[['feature1','feature2']]); y = np.asanyarray(train[['predict_me']])

regr = sklearn.linear_model.LinearRegression(); regr.fit(x,y); n,m = regr.intercept_, regr.coef_
regr.predict(x); regr.score(x, y); sklearn.metrics.r2_score(x,y) # Explained variance score and R2 score: ranges from [-inf worse, +1 best] https://stackoverflow.com/questions/24378176/python-sci-kit-learn-metrics-difference-between-r2-score-and-explained-varian

neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors = k).fit(x_train, y_train)
predictions = neigh.predict(x_test); sklearn.metrics.accuracy_score(y_test, predictions) # Jaccard score
mean_accuracies = [metrics.accuracy_score(y_test, KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train).predict(x_test)) for k in range(1,9+1)]; optimal_k_value = mean_accuracies.argmax()+1

sklearn.tree.DecisionTreeClassifier(criterion="entropy", max_depth=4).fit(x_train,y_train, sample_weight=sklearn.utils.class_weight.compute_sample_weight('balanced', y_train)).predict(x_test)
sklearn.metrics.accuracy_score(y_test, predictions) # Jaccard score
sklearn.tree.plot_tree(fitted_decission_tree); plt.show() # install the "pydotplus" and "graphviz" libraries
sklearn.metrics.roc_auc_score(y_test, fitted_decision_tree.predict_proba(X_test)[:,1])

probabs = sklearn.svm.LinearSVC(class_weight='balanced', loss="hinge", fit_intercept=False).fit(x_train, y_train).decision_function(x_test)
sklearn.metrics.roc_auc_score(y_test, probabs)
sklearn.metrics.hinge_loss(y_test, probabs)

pickle.dumps(model)

sklearn.tree.DecisionTreeRegressor(criterion = "mse", max_depth=8) .fit(x_train, y_train) .predict(x_test) # max_depth, min_samples_split, min_samples_leaf, max_features
regression_tree.score(x_test, y_test); regression_tree.predict(x_test) # R^2 value

snapml.DecisionTreeClassifier(max_depth=4, n_jobs=4).fit(x_train, y_train, sample_weight=w_train) # CPU n_jobs=4 vs GPU use_gpu=True
snapml.SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False).fit(x_train, y_train) # CPU n_jobs=4 vs GPU use_gpu=True
snapml.DecisionTreeRegressor(max_depth=8, random_state=45, n_jobs=4).fit(x_train, y_train).predict(x_test) # CPU n_jobs=4 vs GPU use_gpu=True

sklearn.linear_model.LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train) # C parameter = 1/regularization strength
logistic_regression.predict(x_test); logistic_regression.predict_proba(x_test); label=np.argmax(predicted_probabilities,axis=1) # predict returns label (either 0 or 1) whereas predict_proba returns [probability for 0, probability for 1]
sklearn.metrics.accuracy_score(y_test, predictions)
sklearn.metrics.jaccard_score(y_test, predictions, pos_label=0) # jaccard index = size intersection/size union
sklearn.metrics.confusion_matrix(y_test, predictions, labels=[1,0]); sklearn.metrics.classification_report(y_test, predictions)
# precission=how reliable is it that the prediction is correct; recall=how many of the actual labels is it detecting; F1 score=harmonic average of precission and recall
sklearn.metrics.log_loss(y_test, prediction_probabilities)

sklearn.svm.SVC(kernel='rbf', gamma=0.001, C=100.) .fit(x_train, y_train) .predict(x_test) # Linear, Polynomial, Radial basis function (RBF), Sigmoid
sklearn.svm.SVC(probability=True) .fit(x_train, y_train) .predict_proba(x_test)
sklearn.svm.SVC(decision_function_shape='ovo') .fit(x_train, y_train) .decision_function(x_test) # For multiple classes: ovo=one.vs.one ovr=one.vs.rest BUT internally always uses ONE vs ONE strategy
sklearn.metrics.confusion_matrix(y_test, predictions, labels=["labelname","otherlabel"])
sklearn.metrics.classification_report(y_test, predictions)
sklearn.metrics.f1_score(y_test, predictions, average='weighted'); sklearn.metrics.jaccard_score(y_test, predictions, pos_label="which label mean POSITIVE DETECTION")

model = sklearn.cluster.KMeans(init = "k-means++", n_clusters = 4, n_init = 12) .fit(datapoint_coordiantes) # init=which initial condition chooser, n_init=how many different initial conditions test and choose the best among them, n_clusters=k value
model.cluster_centers_; model.labels_ # labels=which cluster has been asigned to each training datapoint



