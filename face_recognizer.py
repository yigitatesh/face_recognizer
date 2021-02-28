from keras.models import load_model
import cv2
import os
import numpy as np
import pickle
import time

import dlib
from imutils import face_utils

from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

## CHANGABLE VARIABLES ##
# threshold value evaluates whether a face is known or unknown
THRESHOLD = 0.8

# USE FULL POWER OF GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# LOAD PRETRAINED SVM MODEL
svm_filename = "svm_2_faces.sav"
svm = pickle.load(open(svm_filename, "rb"))

# Face Labels
# face_labels = ["helin", "yigit"]
face_labels = os.listdir("face_data/images")

# LOAD PRETRAINED FACENET MODEL
facenet_model = load_model("model/facenet_keras.h5")

# LOAD FACE DETECTOR
detector = dlib.get_frontal_face_detector()

# HELPER PREDICTION FUNCTIONS
def face_to_embedding(image, facenet_model, size=(160, 160)):
    """converts a BGR face image to vector embedding of the image"""
    ## preprocess image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, size)
    # standardize image
    image = image.astype(np.float32)
    mean = image.mean()
    std = image.std()
    image = (image - mean) / std
    # get face embedding
    image = np.expand_dims(image, axis=0)
    embedding = facenet_model.predict(image)
    # normalize embedding
    embedding = Normalizer(norm="l2").transform(embedding)
    
    return embedding

def predict_show_image(image_path, facenet_model=facenet_model, svm=svm,
                      detector=detector):
    # read image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (800, 600))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # find faces
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        # make predictions
        face = image[y1:y2, x1:x2, :]
        embedding = face_to_embedding(face, facenet_model)
        pred_probas = np.squeeze(svm.predict_proba(embedding))
        pred = pred_probas.argmax()
        confidence = pred_probas.max()
        
        # thresh prediction
        if confidence > THRESHOLD:
            # put the name on the face
            name = face_labels[pred]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, "name: {}".format(name),
                       (x1, y1-40), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "confidence: {}".format(str(confidence)[:4]),
                       (x1, y1-20), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            # unknown face
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, "unknown face",
                       (x1, y1-20), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def face_recognizer_camera():
    cap = cv2.VideoCapture(0)

    fps = 0
    last_time = time.time()

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)

        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
            
            # make predictions
            face = img[y1:y2, x1:x2, :]
            embedding = face_to_embedding(face, facenet_model)
            pred_probas = np.squeeze(svm.predict_proba(embedding))
            pred = pred_probas.argmax()
            confidence = pred_probas.max()
            
            # thresh prediction
            if confidence > THRESHOLD:
                # put the name on the face
                name = face_labels[pred]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, "name: {}".format(name),
                           (x1, y1-40), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, "confidence: {}".format(str(confidence)[:4]),
                           (x1, y1-20), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                # unknown face
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, "unknown face",
                           (x1, y1-20), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("img", img)
        
        # fps
        fps += 1
        if time.time() - last_time > 1:
            print("FPS: {}".format(fps))
            fps = 0
            last_time = time.time()
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()

def main():
    run = True
    while run:
        # inform user
        print("\nType c for detection in webcam camera.")
        print("Type i for detection in image.")
        print("Type q to quit.")

        # get the decision
        decision = input("Type: ")

        # process the decision
        if decision == "q":
            run = False
            continue

        elif decision == "c":
            face_recognizer_camera()

        elif decision == "i":
            print("Type the path to the image.")
            print("For example: face_data/test_images/img.jpg")
            image_path = input("Type the path: ")
            image = cv2.imread(image_path)
            if image is None:
                print("Wrong image path!")
                continue
            predict_show_image(image_path)

        else:
            print("Wrong command!")


if __name__ == "__main__":
    main()
