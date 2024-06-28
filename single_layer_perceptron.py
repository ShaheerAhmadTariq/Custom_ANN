import numpy as np
import time
inputs = [[1,1], [1,0], [0,1], [0,0]]
outputs = [1,0,0,0]
learning_rate = 0.05

# initialize weights with random values
weights = np.random.rand(len(inputs[0]))
epochs = 100
total_epochs = 0
start_time = time.time()
for i in range(len(inputs)):
    input_layer = np.array(inputs[i])
    output_layer = np.array(outputs[i])
    for _ in range(epochs):
        total_epochs += 1
        weighted_sum = np.dot(input_layer, weights)
        activation_output = 1/(1+np.exp(-weighted_sum))

        if activation_output > 0.5:
            predicted_output = 1
        else:
            predicted_output = 0

        if predicted_output == output_layer:
            # print("Correct prediction")
            break
        else:
            # print("Incorrect prediction")
            for j in range(len(weights)):
                delta_weights = learning_rate * (output_layer - predicted_output) * input_layer[j]
                weights[j] = weights[j] + delta_weights
            # print("Updated weights: ", weights)
print("Total epochs took: ", total_epochs)
print("Time taken: ", time.time() - start_time)
print("Final weights: ", weights)