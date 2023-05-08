# Sequential data: time series, genes, weather...
# Recurrent Neural Networks: mantain a state, context or memory
# RNN one-to-many: single input, sequence output (captioning of image)
# RNN many-to-one: sequence input, single output (summarize comment into single word)
# RNN limitations include 1) computationally expensive 2) Vanishing Gradient problem 3) Exploding Gradients

# Long Short-Term Memory model (LSTM) https://towardsdatascience.com/lstm-networks-a-detailed-explanation-8fae6aefc7f9
# 1. Forget gate
cell_long_term_state = cell_long_term_state * sigmoid(linear_layer(current_input, cell_hidden_state))
# 2. Input gate
cell_long_term_state += tanh(linear_layer(current_input, cell_hidden_state)) * sigmoid(linear_layer(current_input, cell_hidden_state))
# 3. Output gate
cell_hidden_state = sigmoid(linear_layer(current_input, cell_hidden_state)) * tanh(cell_long_term_state)
# 4. Prediction
prediction = linear_layer(cell_hidden_state)

###
# CODE
###
import tensorflow as tf

inputs = tf.constant(inputs, dtype=tf.float32)
inputs = tf.reshape(inputs,
                    shape=(batch_size, elements_in_sequence, features_of_each_element),
)

# Single LSTM
number_of_features_of_output = 4
lstm = tf.keras.layers.LSTM(number_of_features_of_output, return_sequences=True, return_state=True)
cell_long_term_state, cell_hidden_state = lstm.states
whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)

# Multiple LSTM layers
stacked_cells = tf.keras.layers.StackedRNNCells([
    tf.keras.layers.LSTMCell(4),
    tf.keras.layers.LSTMCell(5),
])
lstm_layer = tf.keras.layers.RNN(stacked_cells, return_sequences=True, return_state=True)
output, final_memory_state, final_carry_state  = lstm_layer(inputs)

###
### Natural Language processing
###
# Sentence -> sequence of words -> recurrent neural network

# WORD EMBEDDING: tf.get_variable("embedding", [vocab_size: 200])
# Randomly intialized, but TRAINABLE: words used in a similar
# context are updated so they have similar vector representations

# Architecture:
# Sequence -> Embedding matrix -> layers with LSTM cells stacked -> Softmax (max probability, decode embedding)

# The Penn Treebank (PTB for short) is a word dataset
# commonly used as a benchmark dataset for Language Modelling.

# Code adapted from the PTBModel example bundled with the TensorFlow source code. (https://github.com/tensorflow/models)

words_per_sentence = 20
batch_size = 30 # sentences per batch-size, where each sentence has 20 words
vocab_size = 10_000
embeding_vector_size = 200
max_grad_norm = 5.

model = tf.keras.models.Sequential([    

    tf.keras.layers.Embedding( # consider using: with tf.device("/cpu:0"):
        vocab_size,
        embeding_vector_size,
        batch_input_shape=(batch_size, words_per_sentence),
        trainable=True,
        name="embedding_vocab"
    ),  # [10000x200]

    tf.keras.layers.RNN(
        tf.keras.layers.StackedRNNCells([
            tf.keras.layers.LSTMCell(256),
            tf.keras.layers.LSTMCell(128)
        ]),
        [batch_size, words_per_sentence], return_state=False, stateful=True, trainable=True
    ),
    # consider intializing with RNN_layer.inital_state = self.tf.Variable( tf.zeros([batch_size,embeding_vector_size]), trainable=False)

    tf.keras.layers.Dense(self.vocab_size),

    tf.keras.layers.Activation('softmax'),
])

optimizer = tf.keras.optimizers.SGD(lr=0.5, clipnorm=max_grad_norm)

model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    optimizer=optimizer,
)

train_data, valid_data, test_data, _, _ = reader.ptb_raw_data("data/simple-examples/data/")
how_many_words_in_training_data = len(train_data)
iterations_per_epoch = ((how_many_words_in_training_data // batch_size) - 1) // words_per_sentence

for epoch in range(15): # epoch
    start_time = time.time()
    cost_per_epoch = 0.

    model.reset_states()

    for iterations, (x, y) in enumerate(reader.ptb_iterator(train_data, batch_size, words_per_sentence)): # mini-batch
        weights = model.trainable_variables
        with tf.GradientTape() as tape:
            output_words_prob = model(x) # forward pass
            loss_vector  = tf.keras.losses.sparse_categorical_crossentropy(y, output_words_prob)
            average_loss = tf.reduce_sum(loss_vector / batch_size) # average across batch
        
        gradients = tape.gradient(average_loss, weights)
        gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
        optimizer.apply_gradients(zip(gradients, weights))

        cost_per_epoch += average_loss

        how_many_sentences_gone_through_so_far = iterations*batch_size

        perplexity_per_epoch_so_far = np.exp(cost_per_epoch / how_many_sentences_gone_through_so_far)
        words_per_second = words_per_sentence * how_many_sentences_gone_through_so_far / (time.time() - start_time)

    perplexity_per_epoch = np.exp(cost_per_epoch / how_many_sentences_gone_through_so_far) # accuracy for language models, lower is better
