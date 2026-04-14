import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np 
train_data = tf.keras.preprocessing.image_dataset_from_directory("//Users/mac/Downloads/train", image_size=(48,48), batch_size=32, color_mode="grayscale")#we load the training data from a directory using the image_dataset_from_directory function. We specify the path to the directory containing our training images, set the image size to (48, 48), and the batch size to 32. We also set color_mode to "grayscale" since our images are in grayscale.
test_data=tf.keras.preprocessing.image_dataset_from_directory("/Users/mac/Downloads/train", image_size=(48,48), batch_size=32, color_mode="grayscale")
labels = []

for images, lbls in train_data:
    labels.extend(lbls.numpy())

labels = np.array(labels)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))


for images,lables in train_data:
    print(images.shape)
    print(lables)
    break    
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
model = tf.keras.models.Sequential([
    data_augmentation,
    
    tf.keras.layers.Rescaling(1./255, input_shape=(48,48,1)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(train_data.class_names), activation='softmax')
])
print(train_data.class_names)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_data,
    epochs=50,
    validation_data=test_data,
    class_weight=class_weights
)
history.history['accuracy']
history.history['val_accuracy']
print(train_data.class_names   )
model.save("face_detection_model.h5" )