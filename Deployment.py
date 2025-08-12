import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
from collections import deque, Counter

model = YOLO(r"C:\Users\moham\Downloads\best_model.pt")  # Replace with your model

st.set_page_config(page_title="Sign Language Detection", layout="wide")
st.title("Real-Time Sign Language Detection with YOLO")

run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

max_buffer_size = 5
predictions_buffer = deque(maxlen=max_buffer_size)


while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to read from webcam.")
        break

    results = model.predict(frame, conf=0.1, stream=True)

    # Draw boxes on the frame
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if len(label) == 1:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])

                predictions_buffer.append((label, conf))

    if predictions_buffer:

        letter_scores = {}
        for label, conf in predictions_buffer:
            if label not in letter_scores:
                letter_scores[label] = []
            letter_scores[label].append(conf)

        best_letter = max(letter_scores.items(), key=lambda x: np.mean(x[1]))[0]

        cv2.putText(frame, f'Most confident letter: {best_letter}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()