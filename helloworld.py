import tensorflow as tf

mnist = tf.keras.datasets.mnist

#LOADING DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


#BUILDING MODEL
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#TRAINING MODEL
model.fit(x_train, y_train, epochs=5)

#EVALUATING (TESTING) MODEL
model.evaluate(x_test,  y_test, verbose=2)

#PROBABILITY DISTRIBUTION INSTEAD
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

#TESTING
probability_model(x_test[:5])
