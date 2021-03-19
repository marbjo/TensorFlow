import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

#LOADING DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#BUILDING MODEL
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), #Input layer, normalizes (Flatten)
  tf.keras.layers.Dense(128, activation='relu'),
  #tf.keras.layers.Dense(128, activation='relu'), #Two layers (probably overkill)
  tf.keras.layers.Dropout(0.2), #Dropout layer prevents overfitting
  tf.keras.layers.Dense(10) #Output layer, 10 nodes (10 numbers, 0-9)
])


#LOSS FUNCTION
#THIS IS BASICALLY DISTANCE IN PARAMETER SPACE, PENALTY FOR WRONG GUESS
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#COMPILING MODEL
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


#Saving model
probability_model.save('num_classifier.model')

#LOADING MODEL
new_model = tf.keras.models.load_model('num_classifier.model')

#PREDICTIONS
predictions = new_model.predict(x_test)


#GRAPHICAL VIEW

x = 20
"""
#x controls which blocks of 10 to show. If you want the 10 first(0-9),
let x=10. If you want the next 10 (10-19), let x=20, and so on.
"""
f, axarr = plt.subplots(2,5)

for i in range(x-10,x-5):
    for j in range(2):
        guess = 'Guess = ' + str(np.argmax(predictions[i+j*5]))
        axarr[j,i%5].imshow(x_test[i+j*5],cmap='gray')
        axarr[j,i%5].set_title(guess,fontsize=20) #Machine guess in subtitle
        axarr[j,i%5].tick_params(
        axis='both',          # Remove ticks and labels
        which='both',
        bottom=False,
        left=False,
        labelleft=False,
        labelbottom=False)

plt.suptitle('Machine guesses on written numbers', fontsize=40)
plt.tight_layout()
#plt.savefig('guess.png',dpi=400)
plt.show()
