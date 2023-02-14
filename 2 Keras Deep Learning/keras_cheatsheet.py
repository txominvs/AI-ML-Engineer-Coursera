dataframe = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/dataframe.csv')
dataframe.head(); dataframe.shape; dataframe.describe()
dataframe.isnull().sum()
features = dataframe[dataframe.columns[dataframe.columns != 'remove_this_column']]; target = dataframe['column_name']
normalized_dataframe = (dataframe - dataframe.mean()) / dataframe.std()

##############
# REGRESSION #
##############

from keras.models import Sequential
from keras.layers import Dense

# create model
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(number_of_features,))) # number_of_features = features_norm.shape[1]
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(features_norm, target, validation_split=0.3, epochs=100, verbose=2)