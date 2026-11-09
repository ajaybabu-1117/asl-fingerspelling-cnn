"""
CNN Model architecture for ASL Hand-Landmark Image Classification.
Designed to take 64x64x1 landmark skeleton images and classify into 26 classes (A-Z).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_asl_cnn(input_shape=(64, 64, 1), num_classes=26):
    """
    Builds a Convolutional Neural Network optimized for hand-landmark images.
    """
    model = models.Sequential([
        # Block 1
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
        layers.Dropout(0.2, name='drop1'),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        layers.Dropout(0.25, name='drop2'),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool3'),
        layers.Dropout(0.3, name='drop3'),

        # Classification Dense Head
        layers.Flatten(name='flatten'),
        layers.Dense(256, activation='relu', name='dense1'),
        layers.BatchNormalization(name='bn_dense'),
        layers.Dropout(0.4, name='drop_dense'),
        layers.Dense(num_classes, activation='softmax', name='output_softmax')
    ], name='ASL_Landmark_CNN')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == '__main__':
    net = build_asl_cnn()
    net.summary()
