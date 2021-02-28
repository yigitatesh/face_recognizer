from keras.models import load_model
import cv2
import os
import numpy as np
import pickle

from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

# FUNCTIONS
def load_faces(directory, size=(160, 160)):
    """Loads face images in the directory and preprocess them"""
    faces = []
    for filename in os.listdir(directory):
        # get path
        path = directory + filename
        # read the face image
        image = cv2.imread(path)
        
        ## preprocess image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size)
        # standardize image
        image = image.astype(np.float32)
        mean = image.mean()
        std = image.std()
        image = (image - mean) / std
        
        faces.append(image)
    
    return faces

def load_face_dataset(directory):
    """Loads faces dataset and labels"""
    X, y, face_labels = [], [], []
    # look all of subdirs (classes)
    for subdir in os.listdir(directory):
        # path of image folder
        path = directory + subdir + "/"
        # load faces in subdir
        faces = load_faces(path)
        # classes
        classes = [subdir] * len(faces)
        # face labels
        face_labels.append(subdir)
        
        # save faces and classes
        X.extend(faces)
        y.extend(classes)
    
    return np.array(X), np.array(y), face_labels

def make_embeddings(faces, facenet_model):
    # get embeddings by facenet model
    embeddings = facenet_model.predict(faces)
    # normalize embeddings
    embeddings = Normalizer(norm="l2").transform(embeddings)
    
    return embeddings

def make_embeddings_dataset(X, y, facenet_model):
    X = make_embeddings(X, facenet_model)
    return X, y

def face_recognition(directory, facenet_model):
    """main training function"""
    # load face dataset
    X, y, face_labels = load_face_dataset(directory)
    # faces to embeddings
    X, y = make_embeddings_dataset(X, y, facenet_model)
    # create svm model
    svm = SVC(kernel="linear", probability=True).fit(X, y)
    
    return svm, X, y, face_labels

# LOAD PRETRAINED FACENET MODEL
facenet_model = load_model("model/facenet_keras.h5")

# LOAD IMAGES AND CREATE SVM MODEL
svm, X, y, face_labels = face_recognition("face_data/images/", facenet_model)

# SAVE THE MODEL
filename = "svm_2_faces.sav"
pickle.dump(svm, open(filename, "wb"))
