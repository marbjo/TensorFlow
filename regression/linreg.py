import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

#Creating "random increasing" data
seed = 1
rng = default_rng(seed)
X = rng.uniform(low=0, high=10, size=15)
Y = rng.uniform(low=0, high=5, size=15)
X = np.sort(X)
Y = X + Y

# Learning parameters
learning_rate = 0.01
training_steps = 1000

# Only for visuals
display_step = 100
plt.style.use('bmh') #For more colors

# Weight and Bias. Initial guess intentionally bad to show progress.
# A more natural choice would be to initialize with rng.standard_normal()
W = tf.Variable(10., name="weight")
b = tf.Variable(-20., name="bias")


# Linear regression
def linear_regression(x):
    return W*x + b

# Mean square error
def mean_square(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Stochastic Gradient Descent Optimizer
optimizer = tf.optimizers.SGD(learning_rate)

# Optimization process.
def run_optimization():
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)

    # Compute gradients.
    gradients = g.gradient(loss, [W, b])

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))

# Run training for the given number of steps.
for step in range(1, training_steps + 1):
    # Run the optimization to update W and b values.
    run_optimization()

    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))
        plt.plot(X, np.array(W * X + b), '--', label=str(step))

# Graphic display
plt.plot(X, Y, 'ro', markersize=10, alpha=0.5, label='Noisy data')
plt.plot(X, np.array(W * X + b), label='Final fit')
plt.title('Linear regression Wx+b, with 1000 learning steps. Each 100th step of tuning W and b is shown.')
plt.legend()
plt.show()
