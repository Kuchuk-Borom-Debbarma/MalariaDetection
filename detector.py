from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


def create_data_generator(batch_size=32):
    return ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )


def build_model(input_shape=(125, 125, 3)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    # Setup paths
    script_dir = os.path.dirname(__file__)
    base_dir = os.path.join(script_dir, 'malaria_detection_data')
    infected_dir = os.path.join(base_dir, 'Parasitized')
    healthy_dir = os.path.join(base_dir, 'Uninfected')

    # Get file lists
    infected_files = glob.glob(infected_dir + '/*.png') + glob.glob(infected_dir + '/*.jpg')
    healthy_files = glob.glob(healthy_dir + '/*.png') + glob.glob(healthy_dir + '/*.jpg')

    print(f"Found {len(infected_files)} infected samples and {len(healthy_files)} healthy samples")

    # Create DataFrame
    files_df = pd.DataFrame({
        'filename': infected_files + healthy_files,
        'label': ['malaria'] * len(infected_files) + ['healthy'] * len(healthy_files)
    })

    # Split data
    train_df, temp_df = train_test_split(files_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Setup data generators
    BATCH_SIZE = 32
    IMG_DIMS = (125, 125)

    train_datagen = create_data_generator()
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    def generator_from_dataframe(dataframe, datagen, batch_size=32):
        while True:
            batch_files = dataframe.sample(n=batch_size)
            batch_images = []
            batch_labels = []

            for _, row in batch_files.iterrows():
                try:
                    img = cv2.imread(row['filename'])
                    if img is None:
                        continue
                    img = cv2.resize(img, IMG_DIMS)
                    img = datagen.random_transform(img)
                    img = img / 255.0
                    batch_images.append(img)
                    batch_labels.append(1 if row['label'] == 'malaria' else 0)
                except Exception as e:
                    print(f"Error processing image {row['filename']}: {str(e)}")
                    continue

            if batch_images:  # Only yield if we have images
                yield np.array(batch_images), np.array(batch_labels)

    # Create model
    model = build_model()

    # Training callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(script_dir, 'models', 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    # Train model
    steps_per_epoch = len(train_df) // BATCH_SIZE
    validation_steps = len(val_df) // BATCH_SIZE

    history = model.fit(
        generator_from_dataframe(train_df, train_datagen, BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        epochs=25,
        validation_data=generator_from_dataframe(val_df, val_datagen, BATCH_SIZE),
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    # Save final model
    save_path = os.path.join(script_dir, 'models', 'final_model.keras')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)


if __name__ == "__main__":
    main()