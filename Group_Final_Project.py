import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load labeled face images for training the model
def load_lfw_images(folder_path):
    image_data = []
    labels = []
    for root, dirs, files in os.walk(folder_path):
        if len(files) >= 20:
            for file in files:
                file_path = os.path.join(root, file)
                label = os.path.basename(root)
                image_data.append((cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), label))
                labels.append(label)
    return image_data, labels

# Detect and crop face from the input image
def detect_and_crop_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = image[y:y + h, x:x + w]
        return cv2.resize(face, (128, 128))
    else:
        return None

# Normalize pixel values of face images
def normalize_images(faces):
    scaler = StandardScaler()
    faces_normalized = scaler.fit_transform(faces)
    return faces_normalized, scaler

# Apply data augmentation by flipping the input image horizontally
def apply_data_augmentation(image):
    return cv2.flip(image, 1)

# Apply Principal Component Analysis (PCA) to reduce dimensionality
def extract_features(faces_normalized, n_components):
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(faces_normalized)
    return features, pca

# Train SVM classifier
def train_svm(features, labels):
    clf = SVC()
    clf.fit(features, labels)
    return clf

# Preprocess an image before making predictions
def preprocess_image(img, scaler, pca):
    face = detect_and_crop_face(img)
    if face is not None:
        flattened_face = face.flatten()
        normalized_face = scaler.transform([flattened_face])
        transformed_face = pca.transform(normalized_face)
        return transformed_face
    else:
        return None

# Predict the label of a normalized face using the trained SVM model
def predict_label(model, normalized_face):
    if normalized_face is not None:
        prediction = model.predict(normalized_face)
        return prediction[0]
    else:
        return None

# Test the trained model on images from a specified folder
def test_model_on_images(model, scaler, pca, image_folder):
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            file_path = os.path.join(root, file)
            input_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            display_image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            normalized_face = preprocess_image(input_image, scaler, pca)

            if normalized_face is not None:
                label = predict_label(model, normalized_face)
                if label is not None:
                    st.image(display_image_rgb, caption=f"Image: {file_path}, Predicted Label: {label}", use_column_width=True)
                else:
                    st.image(display_image_rgb, caption=f"Image: {file_path}, No Label Found", use_column_width=True)
            else:
                st.image(display_image_rgb, caption=f"Image: {file_path}, No Face Detected", use_column_width=True)

# Streamlit UI
st.title("Facial Recognition Web App")

# Load LFW images and train the model
folder_path = r"C:\Users\Admin\Virtual Machine\Final Project\my_virtual_env\lfw_funneled"
image_data, all_labels = load_lfw_images(folder_path)

faces = []
face_labels = []

for img, label in image_data:
    face = detect_and_crop_face(img)
    if face is not None:
        augmented_face = apply_data_augmentation(face)
        faces.append(face.flatten())
        faces.append(augmented_face.flatten())
        face_labels.extend([label, label])

faces_normalized, scaler = normalize_images(faces)
n_components = 1000
features, pca = extract_features(faces_normalized, n_components)
clf = train_svm(features, face_labels)

# File Upload in Streamlit (Multiple Files)
uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        img_array = np.array(img.convert('L'))
        normalized_face = preprocess_image(img_array, scaler, pca)

        if normalized_face is not None:
            label = predict_label(clf, normalized_face)

            # Display the uploaded image and predicted label
            st.image(img, caption=f"Uploaded Image", use_column_width=True)
            st.write(f"Predicted Label: {label}")
        else:
            st.error(f"No face detected in the uploaded image: {uploaded_file.name}")

if uploaded_folder:
    test_model_on_images(clf, scaler, pca, uploaded_folder)
