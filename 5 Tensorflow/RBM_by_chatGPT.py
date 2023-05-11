import numpy as np

class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.weights = np.random.randn(num_visible, num_hidden)
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def positive_phase(self, visible):
        hidden_activations = np.dot(visible, self.weights) + self.hidden_bias
        hidden_probs = self.sigmoid(hidden_activations)
        return hidden_probs

    def negative_phase(self, visible):
        hidden_activations = np.dot(visible, self.weights) + self.hidden_bias
        hidden_probs = self.sigmoid(hidden_activations)
        hidden_states = np.random.binomial(1, hidden_probs)

        visible_activations = np.dot(hidden_states, self.weights.T) + self.visible_bias
        visible_probs = self.sigmoid(visible_activations)
        visible_states = np.random.binomial(1, visible_probs)

        return hidden_probs, hidden_states, visible_probs, visible_states

    def contrastive_divergence(self, visible_input, learning_rate, k=1):
        positive_hidden_probs = self.positive_phase(visible_input)
        positive_associations = np.dot(visible_input.T, positive_hidden_probs)

        # Perform Gibbs sampling for k steps
        visible_states = visible_input
        for _ in range(k):
            hidden_probs, hidden_states, visible_probs, visible_states = self.negative_phase(visible_states)

        negative_hidden_probs = hidden_probs
        negative_associations = np.dot(visible_states.T, negative_hidden_probs)

        # Update the weights and biases
        batch_size = visible_input.shape[0]
        self.weights += learning_rate * (positive_associations - negative_associations) / batch_size
        self.visible_bias += learning_rate * np.mean(visible_input - visible_states, axis=0)
        self.hidden_bias += learning_rate * np.mean(positive_hidden_probs - negative_hidden_probs, axis=0)
        # The mean value is typically used when performing batch
        # training or mini-batch training in RBMs. In these scenarios,
        # instead of updating the weights and biases based on
        # individual training samples, the updates are computed based
        # on the average gradients computed over a batch of training
        # samples.

# Example training data
training_data = np.array([[0, 1, 1, 0, 0],
                          [1, 0, 0, 1, 1],
                          [1, 0, 1, 1, 0],
                          [0, 0, 1, 0, 1]])

# RBM training
rbm = RBM(num_visible=5, num_hidden=3)

num_epochs = 1000
learning_rate = 0.1
k = 1  # Number of Gibbs sampling steps

for epoch in range(num_epochs):
    rbm.contrastive_divergence(training_data, learning_rate, k)

# Test the RBM
test_input = np.array([[1, 0, 1, 0, 0]])
hidden_probs = rbm.positive_phase(test_input)
print("Hidden probabilities:", hidden_probs)