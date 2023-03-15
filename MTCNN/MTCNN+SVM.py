import os
import pickle
import cv2
import numpy as np
from PIL import Image
import csv

from creme.linear_model import LogisticRegression
from creme.multiclass import OneVsRestClassifier
from creme.preprocessing import StandardScaler
from creme.compose import Pipeline
from creme.metrics import Accuracy
from creme import stream
import argparse

from tensorflow.keras.applications import ResNet50
from keras.applications import imagenet_utils
from mtcnn import MTCNN

# The dataset must be structured as follows:
# [dataset]
# |_[name]
#    |_ photo 1
#    |_ photo 2
# |_[name2]
#    |_ photo 1

directory = 'dataset/CASIA-WebFace'     #   directory dataset

#   CSV output to evaluate system performance as follows:
#   first column: real class (the name of the class corresponds to the name of the directory containing the images)
#   ssecond column: predicted class by the system
csvOut = "outputCSV.csv"        

detector = MTCNN()

from keras_facenet import FaceNet
embedder = FaceNet()

THRESHOLD = 0.00 # first set your threshold 
ids = []

# costruttore modello SVM
model = Pipeline(
    ("scale", StandardScaler()),
    ("learn", OneVsRestClassifier(classifier=LogisticRegression())))

# estrazione dei volti e relative features 
def extraction(imageFrame, name, bs):
    image = imageFrame
    pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # individuazione volti
    results = detector.detect_faces(pixels)
    batchImages = []
    dict ={}
    if len(results) > 0:
        for res in results:
            x1, y1, width, height = res['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            # extract face
            face = pixels[y1:y2, x1:x2]
            # resize 
            image2 = Image.fromarray(face)
            image2 = image2.resize((224, 224))
            face_array = np.asarray(image2)
            
            face_array = np.expand_dims(face_array, axis=0)
            face_array = imagenet_utils.preprocess_input(face_array)
            batchImages.append(face_array)

        batchImages = np.vstack(batchImages)
        embeddings = embedder.embeddings(batchImages)
        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import Normalizer
        in_encoder = Normalizer(norm='l2')
        embNorm = in_encoder.transform(embeddings)
        key = 'feat_'
        for i, el in enumerate(embNorm):
            dict.update({key + str(i): el[i]})
        return [name, dict]

# if it finds a new user, learns incrementally, otherwise predicts the class
def findPerson(features, realName, isNew):
        for el in features:
            if not isNew:
                max_value = 0
                class_ = -1
                if el:
                    preds = model.predict_proba_one(el[1])
                    print(preds)
                    for key, value in preds.items():
                        if value > max_value:
                            max_value = value
                            class_ = str(key)
                with open(csvOut, 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    if max_value > THRESHOLD:
                        row = [realName, ids[int(class_)]]
                    if max_value < THRESHOLD or not el:
                        row = [realName, "Unknown"]
                    print(row)
                    writer.writerow(row)
                fd = open(csvOut, "r")
                d = fd.read()
                fd.close()
                m = d.split("\n")
                s = "\n".join(m[:-1])
                fd = open(csvOut, "w+")
                for i in range(len(s)):
                    fd.write(s[i])
                fd.close()
            # write on file
            if isNew and el:
                print('NEW USER FOUNDED!')
                # incremental learning
                model.fit_one(el[1], el[0])





def runDetection(num, total):
    print("[INFO] program started...")
    batch_size = 32
    contatore = 0
    for filen in os.listdir(directory):
        ids.append(filen)
        dire = directory + "/" + filen
        temp = os.listdir(dire)
        person = []
        for k in range(0, num): # use "num" images for training
            file = dire + "/" + temp[k]
            frame = cv2.imread(file)
            person.append(extraction(frame, contatore, batch_size))
        findPerson(person, None, True)
        person = []
        contatore += 1
        print(contatore)

        for i in range(num, total+num): # use other images for testing
            file = dire + "/" + temp[i]
            frame = cv2.imread(file)
            person.append(extraction(frame, None, batch_size))
            findPerson(person, filen, False)
            person = []

    cv2.destroyAllWindows()

if __name__ == "__main__":

    num = 1 # in this case I use 1 image for training 
    total = 25 # I use the other 25 images for test
    runDetection(num, total)
