import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import time

print("🚀 Starting boosted neural network training for MNIST...")

# Load MNIST dataset (digits 0–9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0–255 to 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for convolutional input
y_train = y_train.astype('int')
y_test = y_test.astype('int')
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Data augmentation to improve generalization
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

# Build improved model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    
    layers.Dense(512, activation='swish'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(256, activation='swish'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(128, activation='swish'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),

    layers.Dense(64, activation='swish'),
    layers.BatchNormalization(),

    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for better training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
]

# Start timing
start_time = time.time()

# Train the model
print("📚 Training with augmentation and callbacks...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    validation_data=(x_test, y_test),
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# End timing
end_time = time.time()
print(f"⏱️ Training took {(end_time - start_time)/60:.2f} minutes")

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# Save the model in native Keras format
model.save('trained_digit_model.keras')
print("💾 Model saved as 'trained_digit_model.keras'")
print("🎉 Training complete! You can now run digit_recognition_app.py")
