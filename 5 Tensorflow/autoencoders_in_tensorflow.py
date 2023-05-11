###
### Autoencoders
###
# UNSUPERVISED unlabeled data
# DETERMINISTIC method with NO random involved, for a STOCHASTIC approach see Restricted Boltzmann Machines 
# Uses:
    # feature extraction, dimensionality reduction, compression, generative...
# Architecture:
    # Input -> Progressively smaller layers -> Compressed image -> Progressively bigger layers -> Output should equal Input

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = y_train.astype('float32') / 255.
y_test = y_test.astype('float32') / 255.

input_size = 28*28 # MNIST data input (img shape: 28*28)
n_hidden_1 = 256
n_hidden_2 = 128
encoding_layer = 32 # encoding bottleneck
class Custom_auto_encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.flatten_layer = tf.keras.layers.Flatten()
        self.encoding_1 = tf.keras.layers.Dense(n_hidden_1, activation=tf.nn.sigmoid)
        self.encoding_2 = tf.keras.layers.Dense(n_hidden_2, activation=tf.nn.sigmoid)

        self.code_layer = tf.keras.layers.Dense(encoding_layer, activation=tf.nn.relu)

        self.decoding_1 = tf.keras.layers.Dense(n_hidden_2, activation=tf.nn.sigmoid)
        self.decoding_2 = tf.keras.layers.Dense(n_hidden_1, activation=tf.nn.sigmoid)
        self.decoding_final = tf.keras.layers.Dense(input_size)

    def encoder(self,x):
        x = self.encoding_1(x)
        x = self.encoding_2(x)
        x = self.code_layer(x)
        return x
            
    def decoder(self, x):
        x = self.decoding_1(x)
        x = self.decoding_2(x)
        x = self.decoding_final(x)
        return x
        
    def call(self, x): # when calling model(inputs) this function gets called 
        y_pred  = self.decoder(self.encoder(x))
        return y_pred

flatten_layer = tf.keras.layers.Flatten()
def cost(y_true, y_pred):
    loss = tf.losses.mean_squared_error(y_true, y_pred) # tf.pow(y_true - yhat, 2)
    cost = tf.reduce_mean(loss)
    return cost

model = Custom_auto_encoder()
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
x_train_dataset = tf.data.Dataset.from_tensors(x_train).batch(50).shuffle(buffer_size=100)

for epoch in range(20):
    for inputs in x_train_dataset:

        inputs = flatten_layer(inputs)

        with tf.GradientTape() as tape:    
            reconstructions = model(inputs)
            loss_value = cost(inputs, reconstructions)

        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
