# Sequential data: time series, genes, weather...
# Recurrent Neural Networks: mantain a state, context or memory
# RNN one-to-many: single input, sequence output (captioning of image)
# RNN many-to-one: sequence input, single output (summarize comment into single word)

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
