# 🤖 PyTorch vs. TensorFlow vs. Keras: A Comparison

### 🏛️ Quick Summary

* **TensorFlow** and **PyTorch** are the two primary, complete machine learning frameworks. Think of them as two different, powerful ecosystems for building and deploying models.
* **Keras** is **not** a separate framework anymore. It is a **high-level API** (a user-friendly way to build models) that is **part of the TensorFlow framework**. When you use `tf.keras`, you are using Keras on top of TensorFlow.

---

### 📖 Definitions

* **PyTorch**: Developed by Meta (Facebook), PyTorch is an open-source machine learning framework known for its simplicity, flexibility, and "Pythonic" nature. It's extremely popular in the **research community** because its dynamic computation graph (Eager Execution) makes it easy to build, change, and debug models on the fly.
* **TensorFlow**: Developed by Google, TensorFlow is an end-to-end, open-source platform for machine learning. It's more than just a library; it's a massive **ecosystem** that includes tools for developing, training, and deploying models on servers, mobile devices, and in web browsers (like TensorFlow Lite and TensorFlow Serving). It is heavily used in **production environments**.
* **Keras**: Keras was originally a standalone, high-level API that could run on top of several backends. It was so popular that Google integrated it directly into TensorFlow 2. It is now the **official high-level API for TensorFlow** (as `tf.keras`). It's designed for rapid experimentation and allows you to build a neural network in just a few lines of code.

---

### 📊 Key Differences

This table compares the two main frameworks (PyTorch and TensorFlow) and shows where Keras fits in.

| Feature | PyTorch | TensorFlow (with Keras) |
| :--- | :--- | :--- |
| **Abstraction Level** | Medium-level. You have full control, but write more code (e.g., manual training loops). | High-level (using `tf.keras`). Very user-friendly and abstracts away a lot of boilerplate code. |
| **Primary Use** | **Research & Prototyping**. Favored for its flexibility and ease of debugging. | **Production & Scalability**. Has a robust, mature ecosystem for deploying models. |
| **Code Style** | Feels very "Pythonic." You define models and loops using standard Python logic. | More declarative. You define a model, `compile` it with settings, and then `fit` it to data. |
| **Debugging** | Easy. Since it's dynamic, you can use standard Python debuggers (like `pdb`) and print statements. | Easy (in TF 2.0). Uses Eager Execution by default, just like PyTorch. |
| **Ecosystem** | Strong and growing (TorchServe, TorchScript). | Extremely mature (TensorBoard, TensorFlow Extended, TF Lite, TF.js) for an end-to-end workflow. |

---

### ⌨️ Code Snippets: Simple Neural Network

The best way to see the difference is to build the exact same simple network in both. Notice how PyTorch requires a manual training loop, while Keras handles it with `.fit()`.

#### PyTorch Example

With PyTorch, you define the model as a Python class and write the training loop yourself, giving you full control over every step.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the Model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Define the layers
        self.layer1 = nn.Linear(784, 128) # Input 784, Hidden 128
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)  # Hidden 128, Output 10
    
    def forward(self, x):
        # Define the forward pass
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Dummy data
inputs = torch.randn(64, 784) # 64 samples, 784 features
labels = torch.randint(0, 10, (64,))

# 2. Initialize model, loss, and optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. Manual Training Loop
print("Starting PyTorch training...")
for epoch in range(2): # loop for 2 epochs
    optimizer.zero_grad()   # Clear old gradients
    outputs = model(inputs) # Forward pass
    loss = criterion(outputs, labels) # Calculate loss
    loss.backward()         # Backpropagation
    optimizer.step()        # Update weights
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

```
### TensorFlow (using the Keras API) Example
With Keras, you use the Sequential API to stack layers. 
The model.compile() and model.TEST-DRIVEN DEVELOPMENT()
commands abstract away the entire training loop.

``` python 
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
model.fit(inputs, labels, epochs=2)



```
✅ Which One Should You Use?
Go with TensorFlow (and Keras): If you are a beginner, want to get a model running quickly, or are focused on deploying models into a real-world application (like a website or mobile app).

Go with PyTorch: If you are in research, need to build complex or custom models, or prefer a more flexible, "Python-first" coding experience.
