import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# Generate the training data
sampleSize = 100
xMin = -1.0
xMax = 1.0
inputs = xMin + (xMax-xMin) * np.random.random((sampleSize,1))
outputs = inputs**2
# Note: we are not normalizing the inputs/outputs in this example

# Define the neural network layers
model = Sequential()
model.add(Dense(5, input_dim = 1, activation = 'sigmoid'))
model.add(Dense(1))

# Get an overview of the neural network setup
model.summary()

# Training
model.compile(loss = 'mean_squared_error', optimizer = optimizers.SGD(learning_rate = 0.01))
history = model.fit(inputs, outputs, epochs = 1000, batch_size = 1, verbose = 0)

# Test neural network prediction
print('Output of trained neural network for input = 1.0: ' + str(model.predict([1.0]))) 

# Plot output for trained neural network
x = np.linspace(xMin, xMax, 1000)
plt.plot(x, x**2, label = 'f(x)')
plt.plot(x, model.predict(x), 'k--', label = 'f_NN(x)')
plt.ylabel('Output')
plt.xlabel('Input')
plt.show()

# Plot training loss
plt.plot(history.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()




