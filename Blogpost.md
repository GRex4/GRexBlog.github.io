# Writing Your First Neural Network

If you're interested in the field of artificial intelligence (AI), you've probably heard a lot about neural networks. These powerful algorithms are at the heart of many AI systems, and they are capable of learning and making predictions based on large amounts of data.

But if you're new to the world of AI, you might be wondering how to actually write a neural network. It can seem like a daunting task, especially if you don't have a background in computer science or machine learning.

Don't worry – writing your first neural network is easier than you might think! In this blog post, we'll go over the basics of neural networks and provide a step-by-step guide to writing your first one.

First, let's start by defining what a neural network is. A neural network is a type of machine learning algorithm that is made up of interconnected "neurons". These neurons are organized into layers, and they are able to learn and make predictions based on the connections between them.

![image.png](attachment:image.png)

Imagine a neuron as a little computer inside your brain. It receives information from other neurons through tiny wires, processes that information, and sends it out to other neurons through more tiny wires. Together, all of these neurons work together to help your brain learn and make decisions. For example, when you see a beautiful flower, the neurons in your eyes send information about the flower to other neurons in your brain. These neurons then work together to help you understand what the flower is and how you should react to it. Neurons are a very important part of how your brain works!

"Similarly, a single neuron in a deep learning model uses an activation function to process input data and pass it on to other neurons. By default, this activation function is a linear equation, represented by the formula "y = mx + c"."

![image.png](attachment:image.png)

### Now that we have covered the basic theory, it's time to start coding!

To write a neural network, you'll need to have a dataset that you want your neural network to learn from. This dataset should be large and varied, and it should include examples of the type of data that you want your neural network to be able to handle.

### Here, we are using the in-built MNIST dataset provided by the Keras library.

#### Import TensorFlow  into your program to get started:


```python
import tensorflow as tf
```

#### Load the MNIST dataset


```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

Once you have your dataset, you'll need to preprocess it to extract the relevant information and prepare it for training. This might involve cleaning and normalizing the data, splitting it into individual words or tokens, and performing other types of preprocessing to extract the relevant features.

We are converting the sample data from integers to floating-point numbers


```python
x_train, x_test = x_train / 255.0, x_test / 255.0
```

#### Build a machine learning model

Next, you'll need to define the structure of your neural network. This will involve deciding how many layers it will have, how many neurons each layer will have, and how the neurons will be connected to each other.

"In this case, we are using the Sequential model provided by Keras in TensorFlow. You can read more about it at https://www.tensorflow.org/overview."


```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

here are some glimpse 
    For input layer input shape is required(No. of features we have)
    For hidden layers input variable are not required and can add much as hidden layer as passable 
    For output no. of nodes must be same as output for ex. In classification it = no. of classes
    and in regression it must be  = 1
    Activation function plays important role in output layer's 

### Structure for input layer

model.add(keras.layers.Dense(128,               #Number of nodes
                        input_shape=(4,), 		#Number of input variables
                        name='Hidden-Layer-1',	#Logical name (optional)
                        activation='relu'))   	#activation function (optional)

### Structure for hidden layers

model.add(keras.layers.Dense(128,               #Number of nodes
                       name='Hidden-Layer-2',	#Logical name (optional)
                       activation='relu'))		#activation function

### Structure for output layers

model.add(keras.layers.Dense(NB_CLASSES,		        #Number of nodes
                             name='Output-Layer',		#Logical name (optional)
                             activation='softmax'))		#activation function

Once you have your neural network structure defined, you can use a machine learning library, such as TensorFlow or PyTorch, to train your neural network on your dataset. This will involve iteratively adjusting the weights of the connections between the neurons in order to minimize the error between the predictions made by the neural network and the actual labels in your dataset.

Before you start training, configure and compile the model using Keras Model.compile. Set the optimizer class to adam, set the loss to the loss_fn function you defined earlier, and specify a metric to be evaluated for the model by setting the metrics parameter to accuracy.

Define a loss function for training using losses.SparseCategoricalCrossentropy, which takes a vector of logits and a True index and returns a scalar loss for each example.


```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```


```python
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```

Use the Model.fit method to adjust your model parameters and minimize the loss:


```python
model.fit(x_train, y_train, epochs=5)
```

    Epoch 1/5
    1875/1875 [==============================] - 16s 6ms/step - loss: 0.2927 - accuracy: 0.9156
    Epoch 2/5
    1875/1875 [==============================] - 12s 6ms/step - loss: 0.1431 - accuracy: 0.9575
    Epoch 3/5
    1875/1875 [==============================] - 10s 6ms/step - loss: 0.1075 - accuracy: 0.9674
    Epoch 4/5
    1875/1875 [==============================] - 10s 6ms/step - loss: 0.0879 - accuracy: 0.9729
    Epoch 5/5
    1875/1875 [==============================] - 11s 6ms/step - loss: 0.0749 - accuracy: 0.9762
    




    <keras.callbacks.History at 0x2719196c130>



"We obtained an accuracy of 0.9762, which means our model can classify objects with an accuracy of 97%."

Once your neural network is trained, you can use it to make predictions on new data. This will involve providing your neural network with a set of input data and allowing it to generate output based on the patterns it has learned from your training dataset.

Overall, writing your first neural network is a challenging but rewarding process that can give you a deep understanding of the capabilities and limitations of these powerful algorithms. With the right tools and resources, anyone can write their own neural network and start exploring the fascinating world of AI.
