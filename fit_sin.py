import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # returns array of random normal distribution
        # not setting the biases to zero seems to help fit the graph more evenly
        self.biases =  np.zeros((1, n_neurons)) 
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

class Activation_Softmax:
    def forward(self, inputs):
        # axis keeps each max calculation to the rows
        # keepdims makes sure the dimensions 
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        # this function is a test to see of all the values add up to one
        # self.output = np.sum(probabilities, axis=1)
        # print(self.output)

class Loss:
    def calculate(self, output, target):
        sample_losses = self.forward(output, target)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, target_true):
        n_samples = len(y_pred)
        return np.mean(np.square(target_true - y_pred))

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, target_true):
        n_samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # passed scalar values
        if len(target_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n_samples), target_true]

        # passed one-hot encoded values
        elif len(target_true.shape) == 2:
            correct_confidences = np.sum((y_pred_clipped * target_true), axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

def generate_sin_data(n_samples, lower_bound, upper_bound):
    # X = (upper_bound) * (np.array([np.random.rand(n_samples)]).T + lower_bound)
    X = np.linspace(lower_bound, upper_bound, n_samples).reshape(-1, 1) 
    target = np.sin(X) 
    return (X, target)

def interpolate(low, high, scalar):
    # as accuracy increases the output will get lower
    scalar_clipped = np.clip(np.array(scalar), 1e-16, 1000)
    # if accuracy is 0 output will be same as input
    delta = high - low
    return low + (delta * scalar_clipped)

# Training data, could be any function
X, target = generate_sin_data(200, 0, 4 * np.pi)

# sets up layers 
hidden_layer_neurons = 32 
dense1 = Layer_Dense(1, hidden_layer_neurons)
activation_relu = Activation_Sigmoid()
dense2 = Layer_Dense(hidden_layer_neurons, hidden_layer_neurons)
dense3 = Layer_Dense(hidden_layer_neurons, 1)
activation_softmax = Activation_Softmax()
loss_function = Loss_MeanSquaredError()

lowest_loss = 1 # high initial number 
accuracy = 0

best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
best_dense3_weights = dense3.weights.copy()
best_dense3_biases = dense3.biases.copy()

noise_scale_low = 0.0001
noise_scale_high = 0.01

accept_worse_low = 1.0001
accept_worse_high = 1.001

mse_accuracy_error = 0.01

iterations_per_frame = 1000

show_realtime = True 
first_time = True

loss_history = []
iteration_history = []


for iteration in range(1000000):
    
    noise_scale = interpolate(noise_scale_low, noise_scale_high, lowest_loss)
    accept_worse = interpolate(accept_worse_low, accept_worse_high, lowest_loss)

    dense1.weights += noise_scale * np.random.randn(1, hidden_layer_neurons)
    dense1.biases += noise_scale * np.random.randn(1, hidden_layer_neurons)
    dense2.weights += noise_scale * np.random.randn(hidden_layer_neurons, hidden_layer_neurons)
    dense2.biases += noise_scale * np.random.randn(1, hidden_layer_neurons)
    dense3.weights += noise_scale * np.random.randn(hidden_layer_neurons, 1)
    dense3.biases += noise_scale * np.random.randn(1, 1)

    # perform a forward pass
    dense1.forward(X)
    activation_relu.forward(dense1.output)
    dense2.forward(activation_relu.output)
    activation_relu.forward(dense2.output)
    dense3.forward(activation_relu.output)
    # activation_softmax.forward(activation_relu.output)
    loss = loss_function.forward(dense3.output, target)
    
    if loss < lowest_loss * accept_worse:
        # set best weight and biases to the current change
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        best_dense3_weights = dense3.weights.copy()
        best_dense3_biases = dense3.biases.copy()
        
        lowest_loss = loss
    else:
        # reset the weights and biases from the change
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
        dense3.weights = best_dense3_weights.copy()
        dense3.biases = best_dense3_biases.copy()
    
    # Displays Graph
    if show_realtime and iteration % iterations_per_frame == 0:
        # calculates accuracy for mean squared error within an error
        absolute_errors = np.abs(dense3.output - target)
        correct_predictions = np.sum(absolute_errors < mse_accuracy_error)
        accuracy = correct_predictions / len(target) * 100
        print("Iteration:, ", iteration)
        print("accuracy:, ", np.round(accuracy, 9))
        print("lowest loss:" , lowest_loss)
        print("loss: ", np.round(loss, 9))
        print(" ")
        plt.figure(1)
        plt.clf()  # Clear the current figure
        plt.plot(X, target, color='blue', label='Training data', alpha=0.5)
        plt.plot(X, dense3.output, color='red', label='Neural Network')
        plt.title("Multi-Layer Perceptron without Back Propagation")
        plt.xlabel("Input")
        plt.ylabel("Output sin(input)")
        plt.legend()
        plt.text(X[0] + 0.2, dense3.output[0], f'Accuracy: {np.round(accuracy, 2)}%, Error: +/-{mse_accuracy_error}')
        plt.pause(0.1)
        
        plt.figure(2)
        plt.clf()
        loss_history.append(loss)
        iteration_history.append(iteration)
        plt.plot(iteration_history, loss_history, color='green', label='Loss per Iteration')
        plt.xlabel("Number of Traning Passes")
        plt.ylabel("Lowest Loss Found")
        plt.text(iteration_history[0] , loss_history[0], f'Loss: {np.round(lowest_loss,10)}')
       

    # gives me time to full open graph window otherwise
    # it starts right away
    if first_time:
        plt.pause(5)
        first_time = False

end_time = time.time()
delta_time = end_time - start_time

# Displays final graph at end
print("Computation Time: ", delta_time)
plt.figure(1)
plt.clf()  # Clear the current figure
plt.plot(X, target, color='blue', label='Training data', alpha=0.5)
plt.plot(X, dense3.output, color='red', label='Neural Network')
plt.title("Multi-Layer Perceptron without Back Propagation")
plt.xlabel("Input")
plt.ylabel("Output sin(input)")
plt.legend()
plt.text(X[0] + 0.2, dense3.output[0], f'Accuracy: {np.round(accuracy, 2)}%, Error: +/-{mse_accuracy_error}')

plt.figure(2)
plt.clf()
loss_history.append(loss)
iteration_history.append(iteration)
plt.plot(iteration_history, loss_history, color='green', label='Loss per Iteration')
plt.xlabel("Number of Traning Passes")
plt.ylabel("Lowest Loss Found")
plt.text(iteration_history[0] , loss_history[0], f'Loss: {np.round(lowest_loss,10)}')
plt.show()
