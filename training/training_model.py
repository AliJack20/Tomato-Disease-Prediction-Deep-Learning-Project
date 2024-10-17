
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import models,layers
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML




BATCH_SIZE = 32
IMAGE_SIZE = (256,256)

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=24
)

class_names = dataset.class_names
#print(class_names)


#Visualisation of an image from the tf dataset

for image_batch, label_batch in dataset.take(1):
    plt.imshow(image_batch[0].numpy().astype("uint8"))
  #  plt.show()


EPOCHS = 50


#Train-Test Split
# 80 (Training)
# 10 (Validation)
# 10 (Testing)