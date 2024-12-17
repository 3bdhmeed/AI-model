from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = '../dataset'

# Data preprocessing and augmentation
# -----------------------------------
def get_data_generators(target_size=(224, 224), batch_size=32):

    # Define preprocessing and augmentation parameters.
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,       # Normalize pixel values
        rotation_range=30,       # Rotate images randomly within 30 degrees
        width_shift_range=0.2,   # Translate image width
        height_shift_range=0.2,  # Translate image height
        zoom_range=0.2,          # Zoom in/out
        horizontal_flip=True,    # Randomly flip horizontally
        validation_split=0.2     # Split data into 80% training and 20% validation
    )

    # Generate Training and Validation Data
    # Specify
    #  subset='training' for training data
    #  subset='validation' for validation data.

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator



