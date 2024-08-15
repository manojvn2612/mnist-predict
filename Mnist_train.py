import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the input data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoding
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

class NN:
    def __init__(self):
        self.weights1 = tf.Variable(tf.random.normal([784, 64], stddev=0.1), name='weights1')
        self.bias1 = tf.Variable(tf.zeros([64]), name='bias1')
        self.weights2 = tf.Variable(tf.random.normal([64, 10], stddev=0.1), name='weights2')
        self.bias2 = tf.Variable(tf.zeros([10]), name='bias2')
    
    def forward(self, x):
        x = tf.cast(x, tf.float32)
        x = tf.reshape(x, [-1, 784])
        layer1 = tf.add(tf.matmul(x, self.weights1), self.bias1)
        layer1 = tf.nn.relu(layer1)
        layer2 = tf.add(tf.matmul(layer1, self.weights2), self.bias2)
        return layer2
    
    def compute_loss(self, labels, logits):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

optimizer = tf.optimizers.Adam(learning_rate=0.01)

def train(x, y, model):
    with tf.GradientTape() as tape:
        logits = model.forward(x)
        loss = model.compute_loss(y, logits)
    gradients = tape.gradient(loss, [model.weights1, model.bias1, model.weights2, model.bias2])
    optimizer.apply_gradients(zip(gradients, [model.weights1, model.bias1, model.weights2, model.bias2]))
    return loss

def predict(x):
    logits = nn.forward(x)
    return np.argmax(np.array(tf.nn.softmax(logits)))
if __name__ == "__main__":
    nn = NN()
    
    epochs = 5
    batch_size = 32

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = len(x_train) // batch_size
        for batch in range(0, len(x_train), batch_size):
            x_batch = x_train[batch:batch + batch_size]
            y_batch = y_train[batch:batch + batch_size]
            loss = train(x_batch, y_batch, nn)
            epoch_loss += loss.numpy()

        print(y_train[0])
        print(predict(x_train[0]))
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")
