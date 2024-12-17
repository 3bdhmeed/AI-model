from data_preprocessing import get_data_generators
from model import build_model
from tensorflow.keras.optimizers import Adam




dataset_path = "../dataset"
train_generator, validation_generator = get_data_generators()

model = build_model(num_classes=train_generator.num_classes)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# Print training and validation accuracy
print("Training accuracy: ", history.history['accuracy'][-1])
print("Validation accuracy: ", history.history['val_accuracy'][-1])


model.save("../saved_models/smart_waste_model.keras")
