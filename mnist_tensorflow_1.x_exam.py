import tensorflow as tf 
import pandas as pd

# Model Parameters
learning_rate = 0.001
num_epochs = 10
batch_size = 100

# Load MNIST dataset using Keras datasets API
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and preprocess data
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Network Parameters
n_input = 784  # MNIST data input (28*28 pixels)
n_hidden1 = 512
n_hidden2 = 256
n_classes = 10  # MNIST total classes (0-9 digits)

# Placeholder → TensorSpec으로 대체하고 함수에서 직접 batch_x, batch_y를 받도록 함

# Weights and biases initialization
weights = {
    'h1': tf.Variable(tf.random.normal([n_input, n_hidden1])),
    'h2': tf.Variable(tf.random.normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.random.normal([n_hidden2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random.normal([n_hidden1])),
    'b2': tf.Variable(tf.random.normal([n_hidden2])),
    'out': tf.Variable(tf.random.normal([n_classes]))
}

# Create model
def multilayer_perceptron(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Define loss and optimizer
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Evaluate model
def get_accuracy(pred_logits, true_labels):
    correct_pred = tf.equal(tf.argmax(pred_logits, 1), tf.argmax(true_labels, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Training loop (TF2)
for epoch in range(num_epochs):
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    avg_loss = 0.
    for batch_x, batch_y in dataset:
        with tf.GradientTape() as tape:
            logits = multilayer_perceptron(batch_x)
            loss = loss_fn(batch_y, logits)
        grads = tape.gradient(loss, list(weights.values()) + list(biases.values()))
        optimizer.apply_gradients(zip(grads, list(weights.values()) + list(biases.values())))
        avg_loss += loss.numpy() / (x_train.shape[0] // batch_size)

    acc = get_accuracy(multilayer_perceptron(x_test), y_test).numpy()
    print("Epoch:", '%02d' % (epoch+1), "Loss:", "{:.4f}".format(avg_loss), "Accuracy:", "{:.4f}".format(acc)+"%")
# Final Test Accuracy
final_logits = multilayer_perceptron(x_test)
final_acc = get_accuracy(final_logits, y_test).numpy()
print("Final Test Accuracy:", final_acc, "%")

# TensorFlow version
print("\nTensorflow:", tf.__version__)

# Student Info
data = {
    '이름': ['김수아'],
    '학번': [2112795],
    '학과': ['통계학과']
}
df = pd.DataFrame(data)
print("\n", df)
