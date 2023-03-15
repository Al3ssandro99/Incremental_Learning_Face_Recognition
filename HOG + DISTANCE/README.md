- Running HOG+DISTANCE.py:
  - First, make sure that the dataset is structured as described in line 10.
  - Insert the dataset path at line 18.
  - Insert the CSV file path that will contain the system output at line 23.
  - The CSV file consists of two columns, and for each test image, the first column contains the real class while the second contains the predicted one.

- General Functioning:
  - For each unknown face, the "incrementalLearning" method is called, which incrementally learns a new face at runtime. The learning phase consists of concatenating the encodings related to the new identity and the corresponding name of the subject to the structure containing the known identities.
  -  Each unknown identity will be assigned an ID that coincides with the name of the directory containing the training images of the subjects.