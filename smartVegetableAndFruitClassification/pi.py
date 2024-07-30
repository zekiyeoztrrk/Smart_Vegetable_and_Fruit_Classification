import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Veri setinin yolu
current_dir = os.path.dirname(os.path.realpath(__file__))
weights_dir = os.path.join(current_dir, 'weights')

# Modeli Oluşturma ve Yükleme
img_height = 128
img_width = 128

def create_resnet50_model(num_classes):
    base_model = ResNet50(weights=None, include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# ResNet50 Modelini Yükleme
class_labels = [
    'berry-black', 'berry-blue', 'berry-cherry', 'berry-cherry-ground', 'berry-grape',
    'berry-kiwi', 'berry-rasp', 'berry-straw', 'fruit-apple', 'fruit-banana',
    'fruit-carambola', 'fruit-coconut', 'fruit-dragon', 'fruit-fig', 'fruit-kiwi',
    'fruit-mango', 'fruit-melon', 'fruit-melon-water', 'fruit-papaya', 'fruit-peachy',
    'fruit-pear', 'fruit-persimmon', 'fruit-pineapple', 'fruit-plum', 'fruit-pomegranate',
    'fruit-quince', 'vegetable-avocado', 'vegetable-beet', 'vegetable-carrot', 'vegetable-cucumber',
    'vegetable-edamame', 'vegetable-eggplant', 'vegetable-green', 'vegetable-mushroom',
    'vegetable-olive', 'vegetable-pepper-bell', 'vegetable-pepper-hot', 'vegetable-potato',
    'vegetable-pumpkin', 'vegetable-radish', 'vegetable-tomato', 'vegetable-zucchini'
]

num_classes = len(class_labels)
resnet50_model = create_resnet50_model(num_classes)
best_model_path = os.path.join(weights_dir, 'ResNet50_best_weights.h5')
resnet50_model.load_weights(best_model_path)
resnet50_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Kameradan Görüntü Alma ve İşleme
def predict_and_count_objects_from_camera(model):
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

        # Görüntüyü ön işle
        img = cv2.resize(image, (img_height, img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Tahmin yap
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = class_labels[predicted_class[0]]
        
        # Görüntüyü gri tonlamaya çevir ve nesneleri tespit et
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        _, threshold_image = cv2.threshold(blurred_image, 60, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        object_count = len(contours)
        
        # Görüntü üzerinde tahmin ve sayıları yazdır
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.putText(image, f'{predicted_label}: {object_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # OpenCV ile göster
        cv2.imshow('Object Detection', image)
        
        # 'q' tuşuna basıldığında döngüden çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        rawCapture.truncate(0)
    
    camera.close()
    cv2.destroyAllWindows()

# Kameradan görüntü alarak tahmin yapma ve nesne sayma
predict_and_count_objects_from_camera(resnet50_model)