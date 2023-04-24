import tensorflow as tf
import cv2

img = cv2.imread('train_spec/down/13.png', cv2.IMREAD_UNCHANGED)

dimensions = img.shape
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
 
print('Image Dimension    : ',dimensions)
print('Image Height       : ',height)
print('Image Width        : ',width)
print('Number of Channels : ',channels)

IMAGE_SHAPE = (369, 496)
TRAINING_DATA_DIR = 'train_spec/'
VALID_DATA_DIR = 'test_spec/'

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
train_generator = datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    shuffle=True,
    target_size=IMAGE_SHAPE,
)
valid_generator = datagen.flow_from_directory(
    VALID_DATA_DIR,
    shuffle=False,
    target_size=IMAGE_SHAPE,
)

def build_model(num_classes):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', 
                           input_shape=(369, 496, 3)),
    tf.keras.layers.Normalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Normalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Normalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
model = build_model(num_classes=8)

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)
print(model.summary())

EPOCHS = 200
BATCH_SIZE = 16
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE // 2,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    validation_steps= valid_generator.samples // BATCH_SIZE // 2,
                    verbose=1,
                    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience = 20, min_delta = 0.000001, restore_best_weights = True)
                    )

model.evaluate(valid_generator)

model.save('my_model')