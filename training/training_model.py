
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


def get_partitioned_datasets(dataset,train_split=0.8, val_split= 0.1, test_split=0.1, shuffle= True, shuffle_size=10000):
    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed=10)

    dataset_len= len(dataset)

    train_len = int(train_split*dataset_len)
    train_dataset= dataset.take(train_len)    
    dataset = dataset.skip(train_len)


    val_len = int(val_split*dataset_len)
    val_dataset= dataset.take(val_len)    
    dataset = dataset.skip(val_len)  

    test_len = dataset_len - train_len - val_len
    test_dataset = dataset.take(test_len)
    
    
    return train_dataset, val_dataset, test_dataset


train_dataset, val_dataset, test_dataset = get_partitioned_datasets(dataset)


#print(len(train_dataset))

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size =tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().shuffle(1000).prefetch(buffer_size =tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().shuffle(1000).prefetch(buffer_size =tf.data.AUTOTUNE)


