from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import random
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from concurrent import futures
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

#dataset from https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria?resource=download

# Load and preprocess data
script_dir = os.path.dirname(__file__)
base_dir = os.path.join(script_dir, 'malaria_detection_data')
infected_dir = os.path.join(base_dir, 'Parasitized')
healthy_dir = os.path.join(base_dir, 'Uninfected')

infected_files = glob.glob(infected_dir + '/*.png') + glob.glob(infected_dir + '/*.jpg') + glob.glob(infected_dir + '/*.tiff')
healthy_files = glob.glob(healthy_dir + '/*.png') + glob.glob(healthy_dir + '/*.jpg') + glob.glob(healthy_dir + '/*.tiff')

# Print number of files found
print("Infected Files Found:", len(infected_files))
print("Healthy Files Found:", len(healthy_files))

files_df = pd.DataFrame({
    'filename': infected_files + healthy_files,
    'label': ['malaria'] * len(infected_files) + ['healthy'] * len(healthy_files)
}).sample(frac=1, random_state=42).reset_index(drop=True)

print(files_df.head())


# Split data
train_files, test_files, train_labels, test_labels = train_test_split(files_df['filename'].values,
                                                                      files_df['label'].values,
                                                                      test_size=0.3, random_state=42)
train_files, val_files, train_labels, val_labels = train_test_split(train_files,
                                                                    train_labels,
                                                                    test_size=0.1, random_state=42)

# Load and resize images in parallel
IMG_DIMS = (125, 125)


def get_img_data_parallel(idx, img, total_imgs):
    img = cv2.imread(img)
    img = cv2.resize(img, dsize=IMG_DIMS, interpolation=cv2.INTER_CUBIC)
    img = np.array(img, dtype=np.float32)
    return img


ex = futures.ThreadPoolExecutor(max_workers=None)
train_data_inp = [(idx, img, len(train_files)) for idx, img in enumerate(train_files)]
val_data_inp = [(idx, img, len(val_files)) for idx, img in enumerate(val_files)]
test_data_inp = [(idx, img, len(test_files)) for idx, img in enumerate(test_files)]

print('Loading Train Images:')
train_data_map = ex.map(get_img_data_parallel,
                        [record[0] for record in train_data_inp],
                        [record[1] for record in train_data_inp],
                        [record[2] for record in train_data_inp])
train_data = np.array(list(train_data_map))

print('\nLoading Validation Images:')
val_data_map = ex.map(get_img_data_parallel,
                      [record[0] for record in val_data_inp],
                      [record[1] for record in val_data_inp],
                      [record[2] for record in val_data_inp])
val_data = np.array(list(val_data_map))

print('\nLoading Test Images:')
test_data_map = ex.map(get_img_data_parallel,
                       [record[0] for record in test_data_inp],
                       [record[1] for record in test_data_inp],
                       [record[2] for record in test_data_inp])
test_data = np.array(list(test_data_map))

# Scaling and encoding labels
train_imgs_scaled = train_data / 255.0
val_imgs_scaled = val_data / 255.0

le = LabelEncoder()
train_labels_enc = le.fit_transform(train_labels)
val_labels_enc = le.transform(val_labels)

# Model definition
BATCH_SIZE = 64
EPOCHS = 25
INPUT_SHAPE = (125, 125, 3)

inp = tf.keras.layers.Input(shape=INPUT_SHAPE)
conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inp)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

flat = tf.keras.layers.Flatten()(pool3)
hidden1 = tf.keras.layers.Dense(512, activation='relu')(flat)
drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
hidden2 = tf.keras.layers.Dense(512, activation='relu')(drop1)
drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)
out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

model = tf.keras.Model(inputs=inp, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks and training
logdir = '/content/logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
callbacks = [reduce_lr, tensorboard_callback]

history = model.fit(
    x=train_imgs_scaled, y=train_labels_enc,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(val_imgs_scaled, val_labels_enc),
    callbacks=callbacks,
    verbose=1
)

# Save the model
save_path = os.path.join(script_dir, 'models', 'basic_cnn.keras')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)
