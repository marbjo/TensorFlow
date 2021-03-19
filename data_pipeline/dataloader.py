import numpy as np
import random
import requests
import string
import tarfile
import tensorflow as tf

# Download Titanic dataset (in csv format).
d = requests.get("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/titanic_dataset.csv")
with open("titanic_dataset.csv", "wb") as f:
    f.write(d.content)


# Load Titanic dataset.
# Original features: survived,pclass,name,sex,age,sibsp,parch,ticket,fare
# Select specific columns: survived,pclass,name,sex,age,fare
column_to_use = [0, 1, 2, 3, 4, 8]
record_defaults = [tf.int32, tf.int32, tf.string, tf.string, tf.float32, tf.float32]

# Load the whole dataset file, and slice each line.
data = tf.data.experimental.CsvDataset("titanic_dataset.csv", record_defaults, header=True, select_cols=column_to_use)
# Refill data indefinitely.
data = data.repeat()
# Shuffle data.
data = data.shuffle(buffer_size=1000)
# Batch data (aggregate records together).
data = data.batch(batch_size=2)
# Prefetch batch (pre-load batch for faster consumption).
data = data.prefetch(buffer_size=1)

for survived, pclass, name, sex, age, fare in data.take(1):
    print(survived.numpy())
    print(pclass.numpy())
    print(name.numpy())
    print(sex.numpy())
    print(age.numpy())
    print(fare.numpy())
