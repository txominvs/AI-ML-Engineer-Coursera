# Restricted Boltzmann Machines
    # Autoencoder, UNSUPERVISED unlabeled data
    # STOCASTIC method: random() involved
    # Shallow NN with 2-layers: input and hidden
    # Input -> First layer -> Hidden layer with less neurons (dimensionality reduction) -> Reconstruct input
# Advantages:
    # 1) Collaborative Filtering: guess missing values
    # 2) Automatic feature/pattern extraction
    # 3) More efficient for dimensionality reduction that Principal Component Analysis
# An essential component for Deep Belief Networks
# Steps:
    # Forwards pass: set hidden neuron to 1 with probability = sigmoid( WEIGHTS_MATRIX |input> + |bias_1> )
    # Reconstruction: set input neuron 1 to with probability = sigmoid( <hidden| WEIGHTS_MATRIX + <bias_2| )
# Training:
    # Gibbs Sampling: forward pass + reconstruction
    # Contrastive Divergence:
        # change in WEIGHTS = learning_rate * [ |input><hidden| - |old input><old hidden| ]
        # change in BIAS 1 = learning_rate * [ |old input> - |input> ]
        # change in BIAS 2 = learning_rate * [ |old hidden> - |hidden> ]

import tensorflow as tf

(training_x, training_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
flatten_layer = tf.keras.layers.Flatten(dtype='float32')
training_x = flatten_layer(training_x/255.0)
training_y = flatten_layer(training_y/255.0)
train_ds = tf.data.Dataset.from_tensor_slices((training_x, training_y)).batch(batch_size=200)

alpha_learning_rate = 0.01

visible_bias = tf.Variable(tf.zeros([784]), tf.float32)
hidden_bias = tf.Variable(tf.zeros([50]), tf.float32)
weights_matrix = tf.Variable(tf.zeros([784,50]), tf.float32)
v0_state = tf.Variable(tf.zeros([784]), tf.float32)

for epoch in range(1):
    for batch_x, batch_y in train_ds:
        for sample in batch_x:
            v0_state = sample

            for repeat_optimization_for_each_sample in range(1):
                # hidden layer
                h0_prob = tf.nn.sigmoid(tf.matmul([v0_state], weights_matrix) + hidden_bias) # probabilities of the hidden units
                h0_state = tf.nn.relu(tf.sign(h0_prob - tf.random.uniform(tf.shape(h0_prob)))) # stochastic

                # visible layer
                v1_prob = tf.nn.sigmoid(tf.matmul(h0_state, tf.transpose(weights_matrix)) + visible_bias)
                v1_state = tf.nn.relu(tf.sign(v1_prob - tf.random.uniform(tf.shape(v1_prob)))) # stochastic
                
                # hidden layer
                h1_prob = tf.nn.sigmoid(tf.matmul([v1_state], weights_matrix) + hidden_bias) # probabilities of the hidden units
                h1_state = tf.nn.relu(tf.sign(h1_prob - tf.random.uniform(tf.shape(h0_prob)))) # stochastic

                # update weights
                delta_weights_matrix = tf.matmul(tf.transpose([v0_state]), h0_state) - tf.matmul(tf.transpose([v1_state]), h1_state)
                weights_matrix = weights_matrix + alpha_learning_rate * delta_weights_matrix

                # update biases
                visible_bias = visible_bias + alpha_learning_rate * (v0_state - v1_state)
                hidden_bias = hidden_bias + alpha_learning_rate * (h0_state - h1_state) 
                
                # old state = new state
                v0_state = v1_state

            rms_error = tf.reduce_mean(tf.square(sample - v1_state))

