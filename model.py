# -*- coding: utf-8 -*-
import argparse
import json
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation, Dense, Dropout, Flatten, Lambda, Conv2D, MaxPooling2D, BatchNormalization
)
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam
import utils  # 引入自定义模块

kSEED = 5
SIDE_STEERING_CONSTANT = 0.25
NUM_BINS = 23


def batch_generator(images, angles, augment_data=True, batch_size=64):
    """
    Keras Batch Generator to create a generator of training examples for the model.

    :param images: Training image data.
    :param angles: Angle data for images.
    :param batch_size: Batch size of each training run.
    :param augment_data: If the data should be augmented.

    :return: A batch generator.
    """
    batch_images = []
    batch_angles = []
    sample_count = 0

    while True:
        for i in np.random.permutation(images.shape[0]):
            center_path = images.iloc[i]['Center Image']
            angle = float(angles.iloc[i])

            # Load center image
            center_image = utils.load_image(center_path)
            batch_images.append(center_image)
            batch_angles.append(angle)
            sample_count += 1

            if augment_data:
                # Augmentation 1: Flipping the image
                flipped_image = utils.flip_image(center_path)
                batch_images.append(flipped_image)
                batch_angles.append(-1.0 * angle)

            if len(batch_images) >= batch_size:
                print(f"[DEBUG] Yielding batch with {len(batch_images)} images")
                batch_images_np = np.array(batch_images)
                batch_angles_np = np.array(batch_angles)
                print(f"[DEBUG] Batch images shape: {batch_images_np.shape}, Batch angles shape: {batch_angles_np.shape}")
                yield batch_images_np, batch_angles_np
                batch_images = []
                batch_angles = []


def create_model(lr=1e-3, activation='relu', nb_epoch=15):
    """
    End-to-End Learning Model for Self-Driving based off of Nvidia.

    :param lr: Model learning rate.
    :param activation: Activation function to use for each layer.
    :param nb_epoch: Number of epochs to train for.

    :return: A convolutional neural network.
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    model.add(Conv2D(24, (5, 5), padding='same', activation=activation, strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(36, (5, 5), padding='same', activation=activation, strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(48, (5, 5), padding='same', activation=activation, strides=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation=activation, strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation=activation, strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(356, activation=activation))
    model.add(Dense(128, activation=activation))
    model.add(Dense(52, activation=activation))
    model.add(Dense(10, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


if __name__ == '__main__':
    # Parse command-line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lr", help="Initial learning rate",
                           type=float, default=1e-3, required=False)
    argparser.add_argument("--nb_epoch", help="Number of epochs to train for",
                           type=int, default=15, required=False)
    argparser.add_argument("--activation", help="Activation function to use",
                           type=str, default='relu', required=False)
    args = argparser.parse_args()

    if not os.path.exists('models'):
        os.makedirs('models/')

    file_name = 'driving_log.csv'
    columns = [
        'Center Image',
        'Left Image',
        'Right Image',
        'Steering Angle',
        'Throttle',
        'Break',
        'Speed']

    print('[INFO] Loading Data.')
    images, angles = utils.load_data(file_name, columns)

    print('[INFO] Creating Training and Testing Data.')
    X_train, X_val, y_train, y_val = train_test_split(
        images, angles, test_size=0.15, random_state=kSEED)

    print(f"[DEBUG] Number of training samples: {len(X_train)}, validation samples: {len(X_val)}")
    print(f"[DEBUG] Example training image paths: {X_train.iloc[0]}")

    print('[INFO] Preprocessing Images and Data Augmentation.')
    generator_train = batch_generator(X_train, y_train, augment_data=True)
    generator_val = batch_generator(X_val, y_val, augment_data=False)

    print('[INFO] Creating Model.')
    model = create_model(args.lr, args.activation, args.nb_epoch)
    checkpoint = ModelCheckpoint(
        'models/model-{epoch:03d}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='auto')

    print('[INFO] Training Model.')
    steps_per_epoch = max(1, len(X_train) // 64)
    validation_steps = max(1, len(X_val) // 64)

    # Debugging batch generator output
    sample_images, sample_angles = next(generator_train)
    print(f"[DEBUG] Sample images shape: {sample_images.shape}, Sample angles shape: {sample_angles.shape}")

    model.fit(
        generator_train,
        steps_per_epoch=steps_per_epoch,
        epochs=args.nb_epoch,
        validation_data=generator_val,
        validation_steps=validation_steps,
        callbacks=[checkpoint],
        verbose=1
    )

    print('[INFO] Saving Model.')
    model.save_weights('models/model.h5', True)
    model.save('models/model.keras')
    with open('models/model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
