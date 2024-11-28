import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image


#preproccess images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_dir = "NewImages/Infested"

train_data = train_datagen.flow_from_directory(
    train_dir,  # Folder containing your image dataset
    target_size=(224,224),  # Resize images to target size
    batch_size=32,
    class_mode=None,  # Use None if you're just augmenting data (no labels)
    shuffle=True
    )

print(train_data.samples)