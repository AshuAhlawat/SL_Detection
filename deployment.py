import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from resnetarchi import ResidualUnit, model
from streamlit_cropper import st_cropper
from ultralytics import YOLO
st.title('ASL')
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





def load_image(image_file):
    img = Image.open(image_file)
    img = img.convert("RGB")
    img.save("temp.jpg")
    return img

image_file = st.camera_input("Upload Image")

if image_file is not None:
    file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
    st.write("#### Image Uploaded")
    st.write(file_details)
    st.image(load_image(image_file),width=250)

a = st.button("Predict")
realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None
}
aspect_ratio = aspect_dict[aspect_choice]

if image_file:
    img = Image.open(image_file)
    if not realtime_update:
        st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    
    # Manipulate cropped image at will
    cropped_img.save('cropped_temp.jpg')
    st.write("Preview")
    _ = cropped_img.thumbnail((150,150))
    st.image(cropped_img)
    
if a:
    # hand_model = YOLO('best.pt')
    # hand_model.cpu()
    # results = hand_model('temp.jpg')
    # for result in results:
    #     bbox = result.boxes.xyxy.cpu().numpy()
    
    #     try:
    #         bbox = bbox[0]
    #     except:
    #         st.info('Try again could find hand. sorry for inconvenience')
    
    # bbox=[50,50,250,250]
    # img = Image.open('temp.jpg')
    # print(img.size)
    # img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    # st.image(img, caption='Cropped Part')
    img_cv2 = cv2.imread("cropped_temp.jpg")
    # img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

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