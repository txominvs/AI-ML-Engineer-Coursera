# Neuron (Santiago Ramón y Cajal)
- Soma (activation)
- Dentrites (in)
- Axon (out)

# Artificial Neural Network
- Input layer: first
- Hidden layers: middle set of nodes
- Output layer: last
- Node: neuron
- Dense network: every node is connected to all nodes in the previous and next layers

## Learning
### Forward propagation
- Data goes input layer -> hidden layers -> output layer
- Inside each neuron:
    - Information from all neurons in the previous layers is weighted & summed
    - A biased activation function is applied to decide whether to pass the signal or not
### Compute error
### Back propagation
- Update weights to better match the ground truth:
    - error = 1/2 (prediction - truth)^2 = 1/2 ( SIGMOID(w*x + b) - Y)^2 
    - d[error]/d[w] = (prediction - truth) * derivative of SIGMOID * x
### Repeat
- Until: number of epoch or error threshold

> ### PRO TIP! Vanishing Gradient Problem
> **We do NOT use Sigmoid in the hidden layers of the network,** just the output layer. Both the VALUE and the DERIVATIVE of the sigmoid functions are [-1,1]. So, when we Back Propagate, we are multiplying together factors that are less than 1 which results in a learning rate that decreases from the ouput layer deep into the network.
>
> **Earlier layers in the network train too slowly.** The training takes too long and accuracy is compromised.

## Cost funciton
- Loss function
- Regularization

## Gradient Descent
- new params = old params - learning rate * GRADIENT(params) cost
- the step size automatically decreases as we approach the minimum

## Activation functions
- Logistic function = Sigmoid(x) = 1/(1 + exp( - x))
    - Quite flat at infinity: Vanishing Gradient Problem
    - Not negative values: no symmetry around the origin, all values passed to the next neuron are positive
- Hiperbolic tangent tanh(x) = [exp(x)-exp(-x)] / [exp(x)+exp(-x)]
    - Basically a more simmetric sigmoid
    - Still vanishing gradient problem
- Rectified Linear Unit ReLU(x) = zero for negatives identity for positives
    - **BEST most widely used!** in the Hidden layers of the network
- Softmax(x) = exp(x) / normalize all output
    - Output layer of classification problems
    - Probability of each class
- Others:
    - Binary step function: 0 for negatives and 1 for positives
    - Linear function: basically identity, no activation fx
    - Leaky ReLU(x)

# AI frameworks
- TensorFlow (Google)
- Keras (easy, runs on TensorFlow)
- PyTorch (academic, Torch library in Lua)
- Theano (no longer supported)

# Regression with Keras

```python
model = keras.models.Sequential()
model.add(keras.layers.Dense(5, activation='relu', input_shape=(how_many_features,)))
model.add(keras.layers.Dense(5, activation='relu'))
model.add(keras.layers.Dense(1)) # regression = no output activation

model.complile(optimizer='adam', loss='mean_squared_error', metrics=['mae']) # regression = loss MSE
model.fit(x, y)

predictions = model.predict(x)
```

# Classification with Keras

```python
model = keras.models.Sequential()
model.add(keras.layers.Dense(5, activation='relu', input_shape=(how_many_features,)))
model.add(keras.layers.Dense(5, activation='relu'))
model.add(keras.layers.Dense(4, activation='softmax')) # classification = softmax

model.complile(
    optimizer='adam',
    loss='categorical_crossentropy', # classification
    metrics=['accuracy'])

y_onehotencoded = keras.utils.to_categorical(y) # classification

model.fit(x, y_onehotencoded, epochs=10)

predictions = model.predict(x)
```