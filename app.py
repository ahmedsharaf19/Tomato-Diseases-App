import cv2
from ultralytics import YOLO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
import streamlit as st
from PIL import Image
import time

model = YOLO("best.pt")
mobilenet_model = load_model('MobileNet_Model.h5')

custom_class_names = {
    0: 'Early Blight',
    1: 'Healthy',
    2: 'Late Blight',
    3: 'Leaf Miner',
    4: 'Leaf Mold',
    5: 'Mosaic Virus',
    6: 'Septoria',
    7: 'Spider Mites',
    8: 'Yellow Leaf Curl Virus'
}

colors = {
    0: (173, 216, 230),
    1: (144, 238, 144),
    2: (255, 182, 193),
    3: (240, 230, 140),
    4: (221, 160, 221),
    5: (175, 238, 238),
    6: (238, 130, 238),
    7: (255, 222, 173),
    8: (152, 251, 152)
}

st.title("Tomato Disease Detection")
st.write("Upload an image to detect plant diseases using YOLO and MobileNet.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    start_time = time.time()

    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    st.write("Uploaded Image:")
    col1, col2 = st.columns(2)
    col1.image(original_image, caption="Original Image", use_container_width=True)
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    class_names = model.names
    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = box
        class_name = class_names[int(class_id)]
        cropped_img = image[int(y1):int(y2), int(x1):int(x2)]
        cropped_img_resized = cv2.resize(cropped_img, (224, 224))
        cropped_img_resized = preprocess_input(cropped_img_resized)
        cropped_img_resized = np.expand_dims(cropped_img_resized, axis=0)
        mobilenet_preds = mobilenet_model.predict(cropped_img_resized)
        predicted_class_index = np.argmax(mobilenet_preds, axis=-1)[0]
        mobilenet_class = custom_class_names[predicted_class_index]

        label_1 = f"{class_name} (YOLO)"
        label_2 = f"{mobilenet_class} (MobileNet)"

        color = colors[int(class_id)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        label_size_1 = cv2.getTextSize(label_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_size_2 = cv2.getTextSize(label_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        label_x = int(x2) - max(label_size_1[0], label_size_2[0]) - 10
        label_y = int(y2) - 10

        total_height = label_size_1[1] + label_size_2[1] + 10
        cv2.rectangle(image, (label_x, label_y - total_height), (int(x2) - 10, label_y), color, -1)

        cv2.putText(image, label_1, (label_x, label_y - label_size_2[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(image, label_2, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    col2.image(image, caption="Processed Image", use_container_width=True)

    end_time = time.time()
    processing_time = end_time - start_time
    st.write(f"Processing completed in {processing_time:.2f} seconds!")
