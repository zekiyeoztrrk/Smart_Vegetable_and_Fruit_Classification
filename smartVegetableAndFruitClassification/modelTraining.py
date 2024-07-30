import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns

# Path of the dataset
current_dir = os.path.dirname(os.path.realpath(__file__))
train_dir = os.path.join(current_dir, 'data', 'train')
test_dir = os.path.join(current_dir, 'data', 'test')

# Data Preprocessing and Generator Usage
batch_size = 32
img_height = 128
img_width = 128

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = validation_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

num_classes = len(train_generator.class_indices)

# Model Creation Function
def create_model(base_model, num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_model(model, train_generator, validation_generator, model_name, epochs=20, learning_rate=1e-4):
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(f'{model_name}_best_weights.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs, callbacks=[checkpoint, early_stopping])
    return model, history

# Training and Evaluating Models
models = {
    'MobileNetV2': MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3)),
    'EfficientNetB0': EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3)),
    'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
}

histories = {}
results = {}

for model_name, base_model in models.items():
    print(f'Training {model_name}...')
    model = create_model(base_model, num_classes)
    model, history = train_model(model, train_generator, validation_generator, model_name, epochs=20, learning_rate=1e-4)
    histories[model_name] = history

# Upload and Evaluate the Best Models
def evaluate_model(model, test_generator, class_labels):
    loss, accuracy = model.evaluate(test_generator)
    y_pred = np.argmax(model.predict(test_generator), axis=1)
    y_true = test_generator.classes

    report = classification_report(y_true, y_pred, target_names=class_labels)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    print(f'Loss: {loss}, Accuracy: {accuracy}')
    print(f'F1 Score: {f1}, Precision: {precision}, Recall: {recall}')
    print(f'Classification Report:\n{report}')

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

for model_name, base_model in models.items():
    print(f'Evaluating {model_name}...')
    model = create_model(base_model, num_classes)
    model.load_weights(f'{model_name}_best_weights.h5')
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    evaluate_model(model, test_generator, list(test_generator.class_indices.keys()))

# Visualizing Training Results
for model_name, history in histories.items():
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} Loss')
    
    plt.show()

# Determining the best model
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model_accuracy = results[best_model_name]['accuracy']
print(f'Best model: {best_model_name} with accuracy: {best_model_accuracy}')
print(f'Best model details: {results[best_model_name]}')