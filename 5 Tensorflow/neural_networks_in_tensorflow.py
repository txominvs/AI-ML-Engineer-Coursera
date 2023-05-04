import tensorflow as tf

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 # normalize
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10) # make labels one hot encoded

###
# Simple neural network
###

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(50)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(50)

W = tf.Variable(tf.zeros([784, 10], tf.float32))
b = tf.Variable(tf.zeros([10], tf.float32))
flatten = tf.keras.layers.Flatten(dtype='float32')
def model(x):
    x = flatten(x)
    x = tf.matmul(x, W) + b
    x = tf.nn.softmax(x)
    return x

def cross_entropy(y_label, y_pred):
    EPSILON = 1.e-10
    return - tf.reduce_sum( y_label * tf.math.log(y_pred + EPSILON) )

optimizer = tf.keras.optimizers.SGD(learning_rate=0.25)

for epoch in range(10):
    for mini_batch, (x_train_batch, y_train_batch) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            yhat = model(x_train_batch)
            loss = cross_entropy( y_train_batch, yhat)
            grads = tape.gradient( loss , [W, b] )
            optimizer.apply_gradients( zip( grads , [W, b] ) )     

            loss = loss.numpy()
    
    yhat_test = model(x_test)
    test_loss = cross_entropy(y_test, yhat_test).numpy()
    test_correct_prediction = tf.equal(
        tf.argmax(yhat_test, axis=1),
        tf.argmax(y_test, axis=1)
    )
    test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32)).numpy()

###
# Convolutional Neural Network
###
# activation map = feature map = result of applying kernel to image
# ReLU = x if x >= 0 else: 0
# tf.nn.max_pool(features, ksize=[1,2,2,1], strides=[1,2,2,1])
# Fully Connected layer: flatten + dense network

x_image_train = tf.reshape(x_train, [-1,28,28,1])  
x_image_train = tf.cast(x_image_train, 'float32') 

x_image_test = tf.reshape(x_test, [-1,28,28,1]) 
x_image_test = tf.cast(x_image_test, 'float32') 

# creating new dataset with reshaped inputs
train_ds2 = tf.data.Dataset.from_tensor_slices((x_image_train, y_train)).batch(50)
test_ds2 = tf.data.Dataset.from_tensor_slices((x_image_test, y_test)).batch(50)

# Reduce the size of the dataset for quicker evaluation
x_image_train = tf.slice(x_image_train, [0,0,0,0], [10000, 28, 28, 1])
y_train = tf.slice(y_train, [0,0], [10000, 10])

# Define layers of the network
W_conv1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=0.1, seed=0))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs

W_conv2 = tf.Variable(tf.random.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

W_fc1 = tf.Variable(tf.random.truncated_normal([7 * 7 * 64, 1024], stddev=0.1, seed = 2))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs

W_fc2 = tf.Variable(tf.random.truncated_normal([1024, 10], stddev=0.1, seed = 2)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

def CNN_model(x):
    x = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x = tf.nn.conv2d(x, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x = tf.reshape(x, [-1, 7 * 7 * 64]) # Flattening Second Layer

    x = tf.matmul(layer2_matrix(x), W_fc1) + b_fc1
    x = tf.nn.relu(x)

    # With probability rate elements of x are set to 0. The
    # remaining elements are scaled up by 1.0 / (1 - rate),
    # so that the expected value is preserved.
    x = tf.nn.dropout(x, rate=0.5)

    x = tf.matmul(x, W_fc2) + b_fc2
    x = tf.nn.softmax(x)
    return x

def cross_entropy(y_label, y_pred):
    EPSILON = 1.e-10
    return - tf.reduce_sum( y_label * tf.math.log(y_pred + EPSILON) )
optimizer = tf.keras.optimizers.Adam(1e-4)
variables = [ W_conv1,b_conv1, W_conv2,b_conv2, W_fc1,b_fc1, W_fc2,b_fc2]

for epoch in range(1):
    for mini_batch, (x_train_batch, y_train_batch) in enumerate(train_ds2):
        with tf.GradientTape() as tape:
            yhat = CNN_model(x_train_batch)
            loss = cross_entropy( y_train_batch, yhat)
            grads = tape.gradient( loss , variables )
            optimizer.apply_gradients( zip( grads , variables ) )     

            loss = loss.numpy()
    
    yhat = CNN_model(x_image_train)
    loss = cross_entropy(y_train, yhat).numpy()
    correct_prediction = tf.equal(
        tf.argmax(yhat, axis=1),
        tf.argmax(y_train, axis=1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
