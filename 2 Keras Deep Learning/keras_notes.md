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
## Why they took off now
- Advancements (vanishing gradient problem and ReLU), more data available (avoids overfitting), greater computing power
## Types and architectures of Neural Networks
- Shallow: one or few hidden layer
- Deep: many hidden layers & neurons in each layer
### Supervised
- Convolutional Neural Networks (CNN)
    - Specially tailored for images (Computer Vision) -> less parameters than flatten_image() and fully connected network
    - Input_Shape = (height of the image, width of the image, color chanels = 1 grayscale 3 rgb)
    - Convoltional layer = Filter: a **kernel** makes **strides** through the whole image and we save the **dot product** of kernel and image. Finally an optional **ReLU** activation is applied
    - Max Pooling: slide a box (strides) through the image and only keep the largest value in the box
    - Average Pooling: slide a box but save the average value inside the box
    - Finally, flatten() and fully connected
- Recurrent Neural Networks (RNN)
    - Networks with loops: INPUT = input features + output of the network for the last fed input
    - Good for sequences & patterns: text, genomes, perception of time!
    - Long Short-Term Model (LSTM) popular example: image generation, video captions
### Unsupervised
- Autoencoders
    - Data compression algorithm: automatically learns compression functions
    - Noise removal and dimensionality reduction, more advanced than Principal Component Analysis (PCA) because of non-linearity!
    - Automatically tunes its parameters so that Network output (target) resembles Network input (features)
        - My intuition: Encoder (Dense layer with 10, 8, 4) + Decoder (Dense layer with 4, 8, 10)
    - Restricted Boltzman Machines (RBM) popular example: fix imbalanced dataset, estimate missing values, automatic feature extraction

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
> This problem is solved by using the **ReLU activation** in the hidden layers, instead of the **Sigmoid activation**. This was a great advancement in the field.

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
- TensorFlow (most popular in production, Google)
- Keras (easy, Google, runs on TensorFlow)
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

# Convolutional Neural Network
```python
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(16, kernel_size=2, strides=1, activation='relu',
    input_shape=(image_height,image_width,color_channels,) ))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))

model.add(keras.layers.Conv2D(32, kernel_size=2, activation='relu')) # strides = None = 1
model.add(keras.layers.MaxPooling2D(pool_size=2)) # strides = None = pool_size

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(how_many_classes_output, activation='softmax')) # classification
```