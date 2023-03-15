- Running MTCNN+SVM.py:
  - First, make sure that the dataset is structured as described in line 20.
  - Insert the dataset path at line 28.
  - Insert the CSV file path that will contain the system output. The CSV file consists of two columns, and for each test image, the first column contains the real class while the second contains the predicted one.
  - Insert the number of examples to use for training at line 153.
  - Insert the total number of examples to use for testing at line 154.

- General Functioning:
  - For each unknown face, the "fit_one" method is called, which incrementally learns a new face at runtime.
  - Each unknown face will be assigned an incremental ID that coincides with the index of its name. For example, 0 = name1, 1 = name2, and so on. Through a function, the correspondence between the incremental index and the real name that coincides with the name of the directory where the subject's images are located is automatically mapped.