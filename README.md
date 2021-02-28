# Face Recognizer

# Requirements
- keras
- tensorflow-gpu
- cuda and cudnn support
- opencv-python
- imutils
- dlib
- sklearn
- pickle

# How to prepare to run
There are three steps to prepare recognizer run:
1. Extract facenet_keras.rar file
2. Prepare images
3. Run face_recognizer_trainer.py to train svm model

## Extracting facenet model data
Extract "facenet_keras.rar" file in "models" directory.

## Preparing images
There are sample images in face_data/images folder. <br>
You may delete my images. <br>
Crop faces from your images. <br>
For each person: <br>
- Create a subfolder for the person under the directory "face_data/images".
- Name the subfolder as the person's name.
- Put face images of the person to this subfolder.

## Train recognizer model
Just run face_recognizer_trainer.py

# How to Run the Face Recognizer
After preparing images and training svm model, run face_recognizer.py.
If you have already trained svm model and images are same, there is no need to train it again.
