from PIL import Image
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import os
from instaloader import Instaloader, Profile, ProfileNotExistsException

from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import streamlit as st


# Cleanup download directory
def delete_directory_and_contents(directory_path):
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            delete_directory_and_contents(item_path)
    os.rmdir(directory_path)


# Detect objects in image
def get_face_coords(file_path, detections):
  # Load image
  image_data = pyplot.imread(file_path)
  cropped_images = []
  face_coords = [] # Face coordinates
  for detection in detections:
    # Check confidence
    if detection['confidence'] > 0.96:
      x1, y1, width, height = detection['box']
      x2, y2 = x1 + width, y1 + height
      face_coords.append([x1,y1,x2,y2,width,height])
      cropped_images.append(image_data[y1:y2,x1:x2])
  return cropped_images, face_coords


# Predict gender
def predict_gender(cropped_images):
  for image in cropped_images:     
    # Process image
    processed_img = Image.fromarray(image, 'RGB').resize(target_size).convert('RGB')
    processed_img = img_to_array(processed_img) / 255.0
    processed_img = processed_img.reshape(1, img_size, img_size, 3)

    # Predict
    prediction = model.predict(processed_img)[0][0]
    if prediction >= 0.5:
      predictions = ["Male, {:.2f}".format(prediction)] 
    else:
      prediction = 1 - prediction
      predictions = ["Female, {:.2f}".format(prediction)]
  return predictions


# Variables
img_size = 249 
target_size = (img_size, img_size)
model_path = "model.h5"

# Load model
model = tf.keras.models.load_model(model_path)

ig_username = input("Enter your Instagram username for login: ")
ig_password = input("Enter your Instagram password: ")

loader = Instaloader()
loader.login(ig_username, ig_password)

while True:
    username = input("Enter the username (leave blank to exit): ")
    
    if not username:
      break

    loader = Instaloader()

    try:
      loader.download_profile(username, profile_pic_only=True)
      
    except Exception as e:
      print(f"An error occurred: {e} The username '{username}' does not exist. Please try again.")
      continue
    loader.download_profile(username, profile_pic_only=True)

    # path
    download_dir = os.path.join(os.getcwd(), username)
    image_files = [f for f in os.listdir(download_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No image files found.")
        continue
    full_path = os.path.join(download_dir, image_files[0])

    try:
        uploaded_img = Image.open(full_path)
    except IOError:
        print(f"Could not open image file {full_path}.")
        continue
    detector = MTCNN()
    img_array = img_to_array(uploaded_img)
    detected_faces = detector.detect_faces(img_array)

    if not detected_faces:
        print("No faces detected in the image.")
        continue

    cropped_faces, coords = get_face_coords(full_path, detected_faces)
    if cropped_faces:
        gender_predictions = predict_gender(cropped_faces)
        print(gender_predictions)
    else:
        print("No faces were processed for gender prediction.")

    delete_directory_and_contents(download_dir)

