import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Загрузка модели SSD как Keras слоя
ssd_model = hub.KerasLayer("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1", trainable=False)

# Загрузка модели Faster R-CNN как Keras слоя
faster_rcnn_model = hub.KerasLayer("https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1", trainable=False)

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to load the image from the path: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    img /= 255.0
    return img

def detect_objects(model, image):
    # Обработка изображения и получение результата от модели
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    outputs = model(image_tensor)  # Вызов модели

    # Вместо доступа к несуществующему ключу, используем доступные ключи
    detection_boxes = outputs['detection_boxes']
    detection_scores = outputs['detection_scores']
    detection_classes = outputs['detection_class_entities']  # или 'detection_class_labels' в зависимости от того, что вам нужно

    return detection_boxes, detection_scores, detection_classes

def aggregate_results(ssd_results, rcnn_results):
    # Извлечение результатов из выхода модели
    ssd_boxes = ssd_results['detection_boxes'].numpy()
    ssd_scores = ssd_results['detection_scores'].numpy()
    ssd_classes = ssd_results['detection_classes'].numpy()

    rcnn_boxes = rcnn_results['detection_boxes'].numpy()
    rcnn_scores = rcnn_results['detection_scores'].numpy()
    rcnn_classes = rcnn_results['detection_classes'].numpy()

    # Агрегация результатов
    boxes = np.concatenate([ssd_boxes, rcnn_boxes], axis=0)
    scores = np.concatenate([ssd_scores, rcnn_scores], axis=0)
    classes = np.concatenate([ssd_classes, rcnn_classes], axis=0)

    # Сортировка по скорам и взятие топ N результатов
    top_indices = np.argsort(-scores)[:10]
    return boxes[top_indices], scores[top_indices], classes[top_indices]

# Загрузка и обработка изображения
image_path = r"F:\\git\\bora\\Image\\1.jpg"
image = load_image(image_path)

# Детекция с помощью SSD
ssd_results = detect_objects(ssd_model, image)

# Детекция с помощью Faster R-CNN
rcnn_results = detect_objects(faster_rcnn_model, image)

# Агрегирование результатов
final_boxes, final_scores, final_classes = aggregate_results(ssd_results, rcnn_results)
print("Boxes:", final_boxes)
print("Scores:", final_scores)
print("Classes:", final_classes)
