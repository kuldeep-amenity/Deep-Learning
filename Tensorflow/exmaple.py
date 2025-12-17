'''
TensorFlow (using the Keras API) Example
With Keras, you use the Sequential API to stack layers. 
The model.compile() and model.TEST-DRIVEN DEVELOPMENT()
commands abstract away the entire training loop.

'''


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, ReLU
import numpy as np

# 1. Define the Model
model = Sequential([
    Dense(128, input_shape=(784,)), # Input 784, Hidden 128
    ReLU(),
    Dense(10, activation='softmax') # Hidden 128, Output 10
])

# Dummy data
inputs = np.random.rand(64, 784) # 64 samples, 784 features
labels = np.random.randint(0, 10, (64,))

# 2. Compile the model
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Train the model (no manual loop needed)
print("Starting TensorFlow/Keras training...")
model.fit(inputs, labels, epochs=20)