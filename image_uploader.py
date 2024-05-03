import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from resnetarchi import ResidualUnit, model
from ultralytics import YOLO
import cv2
mapper = {
 0:'A',
 1:'B',
 2:'C',
 3:'D',
 4:'E',
 5:'F',
 6:'G',
 7:'H',
 8:'I',
 9:'J',
 10:'K',
 11:'L',
 12:'M',
 13:'N',
 14:'O',
 15:'P',
 16:'Q',
 17:'R',
 18:'S',
 19:'T',
 20:'U',
 21:'V',
 22:'W',
 23:'X',
 24:'Y',
 25:'Z',
 }


image_up = st.file_uploader('upload image', type=['jpg'])

btn = st.button('Upload and Predict')



if btn:
    hand_model = YOLO('best.pt')
    hand_model.cpu()
    img = Image.open(image_up)
    img.save('temp.jpg')
    results = hand_model('temp.jpg')
    for result in results:
        bbox = result.boxes.xyxy.cpu().numpy()[0]
    cropped_img = img.crop((bbox[0], bbox[1], bbox[2],bbox[3]))
    cropped_img = cropped_img.resize((128, 128))
    st.image(img, caption='Cropped Part')
    img.save("cropped_temp.jpg")
    img_cv2 = cv2.imread("cropped_temp.jpg")
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

    # Perform any operations with OpenCV on the image here
    # For example, you can resize the image
    img_cv2_resized = cv2.resize(img_cv2, (224, 224))

    # Make predictions on the processed image
    img_resized = img_cv2_resized / 255.0
    img_resized = img_resized.reshape(1, 224, 224, 3)
    pred = model.predict(img_resized)
    major_index = np.argmax(pred[0])

    st.write(major_index)
    st.info(f'Predicted Class : {mapper[major_index]}') 