# 1) Linear regression and logistic regression
# 2) Convolutional Neural Networks (CNNs)
# 3) Recurrent Neural Networks (RNNs): Sequences, Long-Short Term Memory, language modelling
# 4) Unsupervised learning and Restricted Boltzmann Machines (RBMs): a movie recommendation
# 5) Unsupervised learning and Autoencoders: detecting patterns

import tensorflow as tf

###
### New method
###
assert tf.executing_eagerly() == True

a = tf.constant([1, 2, 3, 4])
b = tf.constant(np.array([5, 6, 7, 8]))
c = tf.Variable([8, 9, 10, 11])
# VARIABLES vs CONSTANT TENSORS:
# 1) Variables can be changed using the .assign() .assign_add() .assign_sub() methods
# 2) Variables are tracked by default by the GradientTape() since tf.Variable(trainable=True)
#    whereas Constants must be manually tracked using GradientTape().watch(constant)
# 3) tf.keras.Sequential([layers]) automatically collets all Variables in model.trainable_variables[]
d = tf.convert_to_tensor([123, 456,])
back_to_numpy = a.numpy()

operations = tf.matmul, tf.add, tf.subtract, tf.nn.sigmoid, ...
dot_product = tf.tensordot(a, b, axes=1)
addition_result = c + 1

# Derivatives
x = tf.Variable(3.0)
with tf.GradientTape() as g:
    y = x * x
(dy_dx,) = g.gradient(y, [x,])
# or equivalently
x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)
    y = x * x
(dy_dx,) = g.gradient(y, [x,])

@tf.function # for a performance boost
def add(first, second):
    c = tf.add(first, second)
    print(c)
    return c
result_of_function = add(a, b)

x = tf.constant(x)
layer = tf.keras.layers.Softmax(); y = layer(x)
# or equivalently
y = tf.nn.softmax(x)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(50).shuffle(buffer_size=100)
for x, y in train_ds: # mini-batch = 50
    ...

###
### Old method
###
from tensorflow.python.framework.ops import disable_eager_execution; disable_eager_execution()
assert tf.executing_eagerly() == False

a = tf.constant([1, 2, 3, 4])
b = tf.constant(np.array([5, 6, 7, 8]))
dot_product = tf.tensordot(a, b, axes=1)

with tf.compat.v1.Session() as sess: # sess = tf.compat.v1.Session()
    output = sess.run(dot_product) # evaluate result of variable
    # sess.close()


###
# Linear Regression
###
df = pd.read_csv("FuelConsumption.csv"); df.head()
train_x, train_y = np.asanyarray(df[['ENGINESIZE']]), np.asanyarray(df[['CO2EMISSIONS']])

weight = tf.Variable(20.0)
bias = tf.Variable(30.2)
def forward(x):
   y = weight*x + bias
   return y

def loss_object(y, train_y):
    return tf.reduce_mean(tf.square(y - train_y))
# or equivalently
loss_object = tf.keras.losses.MeanSquaredError()

learning_rate = 0.01
for epoch in range(200):
    with tf.GradientTape() as tape:
        y_predicted = forward(train_x)
        loss_value = loss_object(train_y, y_predicted)

        gradients = tape.gradient(loss_value, [weight, bias])

        weight.assign_sub(gradients[0] * learning_rate) # weight = weight - learning_rate * d[loss]/d[weight]
        bias.assign_sub(gradients[1] * learning_rate)

###
# Logistic Regression
###
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
iris_y = pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)
train_X = tf.constant(trainX, dtype='float32')
train_Y = tf.constant(trainY, dtype='float32')
test_X = tf.constant(testX, dtype='float32')
test_Y = tf.constant(testY, dtype='float32')

numFeatures, numLabels = train_X.shape[1], train_Y.shape[1]

weights = tf.Variable(tf.random.normal([numFeatures, numLabels], mean=0., stddev=0.01, name="weights"),dtype='float32')
bias = tf.Variable(tf.random.normal([1, numLabels], mean=0., stddev=0.01, name="bias"))

def forward(x):
    apply_weights_OP = tf.matmul(x, weights, name="apply_weights")
    add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 
    activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")
    return activation_OP

learning_rate_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0008, decay_steps=train_X.shape[0], decay_rate=0.95, staircase=True) # learning rate decay
optimizer = tf.keras.optimizers.SGD(learning_rate_scheduler)
loss_object = tf.keras.losses.MeanSquaredLogarithmicError()

def accuracy(y_pred, y_true): # Predicted class is the index of the highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for epoch in range(700):
    with tf.GradientTape() as g:
        pred = forward(train_X)
        loss = loss_object(pred, train_Y)
    gradients = g.gradient(loss, [weights, bias])
    optimizer.apply_gradients(zip(gradients, [weights, bias]))
    
    pred = forward(test_X)
    validation_loss = loss_object(pred, test_Y)
    validation_accuracy = accuracy(pred, test_Y)

# Convolutional NN: automatic feature extraction
# Recurrent NN: sequential data, language translations
# Restricted Boltzman Machine: unsupervised patterns
# Deep Belief Networks: stack of RBMs, image classification
# Autoencoder: unsupervised, feature compression