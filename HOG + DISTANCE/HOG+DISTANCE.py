import glob
import os
import csv

import numpy as np
import face_recognition
import pickle
import cv2

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

data = {"encodings": [], "names": []}
detectionModel = "hog"


def incrementalLearning(enc, data, name):
    i = 0
    for e in enc:
        data["encodings"].append(e)
        data["names"].append(name)


def findPerson(imageFrame, realName, isNew):
    boxes = face_recognition.face_locations(imageFrame,
                                            model=detectionModel)
    encodings = face_recognition.face_encodings(imageFrame, boxes)
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"
        if True in matches and not isNew:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)

        #write on file
        if not isNew:
            with open(csvOut, 'a', encoding='UTF8') as f:
                writer = csv.writer(f)
                row = [realName, name]
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
        #fine scrivi su file

        if isNew:
            print('NEW USER FOUNDED!')
            incrementalLearning(encodings, data, realName)
        names.append(name)
        #(name)



def runDetection():
    print("[INFO] program started...")
    contatore = 0
    for filen in os.listdir(directory): 
        dire = directory + "/" + filen
        temp = os.listdir(dire)
        file = dire + "/" + temp[0] # use first image for training
        frame = cv2.imread(file)
        findPerson(frame, filen, True)
        contatore += 1
        print(contatore)

        for i in range(1, 26): # use other images for testing
            file = dire + "/" + temp[i]
            frame = cv2.imread(file)
            findPerson(frame, filen, False)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    runDetection()
