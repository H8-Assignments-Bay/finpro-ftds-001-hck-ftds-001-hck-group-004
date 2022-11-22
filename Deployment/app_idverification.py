from turtle import width
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2 
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import streamlit as st
from deepface import DeepFace

st.image('soter.png', width=350)
st.title('Project SOTER')
st.header('Quick ID & Background Checking for Credit Risk Analysis')


st.subheader('ID Verification')

def emosion_detector(image):
    face_detector = MTCNN() #Load model pendeteksi wajah (MTCNN)

    img = plt.imread(image) #Load gambar
    faces = face_detector.detect_faces(img) #Deteksi wajah-wajah yang ada di gambar

    for i in range(len(faces)):
        x, y, w, h = faces[i]['box'] #Ambil koordinat area wajah yang terdeteksi
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 6) #Buat persegi panjang yang sesuai dengan koordinat area wajah dengan warna merah (255,0,0) dan ketebalan 6 pt
        crop = cv2.resize(img[y:y+h,x:x+w],(48,48)) #Potong gambar sesuai dengan area wajah dan mengubah ukuran ke 48x48
        cv2.imwrite(f"face_img{i+1}.jpg",crop)


upload_file = st.file_uploader(label='Upload your image')
if upload_file is not None:
    st.image(upload_file)

if st.button('Submit'): 
    if upload_file is not None: 
        emosion_detector(upload_file)
        vk1 = DeepFace.represent(img_path='face_img1.jpg', enforce_detection=False)
        vk2 = DeepFace.represent(img_path='face_img2.jpg', enforce_detection=False)
        CS = cosine_similarity(np.array([vk1]), np.array([vk2]))
        if CS > 0.60:
            st.image('https://www.pngall.com/wp-content/uploads/5/Green-Checklist-PNG-Image.png', width=100)
            st.write('The pictures match')
        else:
            st.image('https://image.similarpng.com/very-thumbnail/2021/06/Cross-mark-icon-in-red-color-on-transparent-background-PNG.png', width=100)
            st.write('The pictures do not match')
    else: 
        st.write('Face Undetected')

