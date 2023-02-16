dataframe = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/dataframe.csv')
dataframe.head(); dataframe.shape; dataframe.describe()
dataframe.isnull().sum()
features = dataframe[dataframe.columns[dataframe.columns != 'remove_this_column']]; target = dataframe['column_name']
normalized_dataframe = (dataframe - dataframe.mean()) / dataframe.std()


model.save('classification_model.h5')
model = keras.models.load_model('classification_model.h5')


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
matplotlib.pyplot.imshow(X_train[0])
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]).astype('float32') # flatten
x_train = x_train / 255 # normalize


model = keras.models.Sequential()
model.add(keras.layers.Dense(50, activation='relu', input_shape=(number_of_features,))) # number_of_features = features_norm.shape[1]
model.add(keras.layers.Dense(50, activation='relu'))


##############
# REGRESSION #
##############

model.add(keras.layers.Dense(1)) # regression -> no activation

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(features_norm, target, validation_split=0.3, epochs=100, verbose=2)

##################
# CLASSIFICATION #
##################

model.add(keras.layers.Dense(num_classes, activation='softmax')) # classification -> softmax

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

y_train = keras.utils.to_categorical(y_train) # one-hot encode outputs

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, verbose=2)

loss_value, *metrics_values = model.evaluate(x_test, y_test, verbose=0)

#########################
# CONVOLUTIONAL NETWORK #
#########################

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train = x_train / 255

y_train = keras.utils.to_categorical(y_train)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, kernel_size=5, strides=1, # default stride = 1
    activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPool2D(pool_size=2, strides=2)) # default stride = pool size

model.add(keras.layers.Conv2D(8, kernel_size=2, activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(y_test, y_test), epochs=10, batch_size=200, verbose=2)

loss_value, *metrics_values = model.evaluate(x_test, y_test, verbose=0)