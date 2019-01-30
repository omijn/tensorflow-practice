import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

mnist_data = tf.keras.datasets.mnist.load_data()

X = mnist_data[0][0]
y = mnist_data[0][1]

dataset_size = X.shape[0]
img_shape = (X.shape[1], X.shape[2])
img_size_flat = img_shape[0] * img_shape[1]
num_classes = 10

# flatten images
X = X.reshape(dataset_size, img_shape[0] * img_shape[1])

# one hot encode labels
y = tf.keras.utils.to_categorical(y, num_classes=num_classes)


def plot_img(img_vecs):
    for img_vec in img_vecs:
        plt.imshow(img_vec.reshape(img_shape))
        plt.show()


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=50000)
Xtest, Xval, ytest, yval = train_test_split(Xtest, ytest, train_size=5000)


X = tf.placeholder(dtype=tf.float32, shape=[None, img_size_flat])
y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
y_cls = tf.placeholder(dtype=tf.int64, shape=[None])

weights = tf.Variable(tf.zeros((img_size_flat, num_classes)))
biases = tf.Variable(tf.ones((num_classes)))
logits = tf.matmul(X, weights) + biases

ypred = tf.nn.softmax(logits)
ypred_cls = tf.argmax(ypred, axis=1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

optimizer = tf.train.AdamOptimizer().minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_cls, ypred_cls), dtype=tf.float32))

num_epochs = 50

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_size = 64
    train_size = len(Xtrain)
    batch_indices = list(range(0, train_size, batch_size))
    if batch_indices[:-1] != train_size:
        batch_indices.append(train_size)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs), end=' : ')
        for i in range(1, len(batch_indices) + 1):
            Xbatch = Xtrain[i - 1:i]
            ybatch = ytrain[i - 1:i]

            sess.run(optimizer, feed_dict={X: Xbatch, y: ybatch})

        acc = sess.run(accuracy, feed_dict={X: Xval, y: yval, y_cls: yval.argmax(axis=1)})
        print("Validation accuracy = {}".format(acc))
