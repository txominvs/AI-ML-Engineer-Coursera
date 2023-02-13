# Neuron (Santiajo Ramón y Cajal)
- Soma (activation)
- Dentrites (in)
- Axon (out)

# Artificial Neural Network
- Input layer: first
- Hidden layers: middle set of nodes
- Output layer: last
- Node: neuron

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

> ### PRO TIP!
> We do *not* use SIGMOID in the hidden layers of the network, just the output layer.
> Both the VALUE and the DERIVATIVE of the sigmoid functions are [-1,1].
> So, when we *Back Propagate*, we are multiplying together factors that are less than 1
> which results in a learning rate that decreases from the ouput layer deep into the network.
> *Earliest layers in the network train too slowly.* The learning rate takes too long and accuracy compromised.

## Cost funciton
- Loss function
- Regularization

## Gradient Descent
- new params = old params - learning rate * GRADIENT(params) cost
- the step size automatically decreases as we approach the minimum

## Activation functions
- Sigmoid(x) = 1/(1 + exp( - x))