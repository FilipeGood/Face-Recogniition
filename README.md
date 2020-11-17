# Face-Recogniition
Face Recognition with OpenCV - Python


This repository contains two projects that recognize faces:

    * The "normal" one simply uses haar features and local binary patterns (openCV) to recognize faces
    * The one in the face-recognition-deep-learning folder uses deep learning to recognize faces

### Running options:
  - python3 face_recognition.py **test_video**  - tests if cv2 is properly installed and if the webcam works
  - python3 face_recognition.py **create_dataset**   - saves pictures taken from the webcam
  - python3 face_recognition.py **train**   - trains and saves the trainer
  - python3 face_recognition.py **predict_webcam**   - opens the webcam and tries to predict the labels of the faces
  - python3 face_recognition.py **predict_screen**   - uses the image stream from the screen.
  
### face-recognition-deep-learning folder:
  - This folder contains the code for face recognition *using deep learning*
  - Running order:
        - extract_embeddings.py - uses a pytorch implementation of FaceNet in order to extract face embeddings (128 features)
        - train_model.py - uses a SVM (sklearn) to train a classifier on the face embeddings features
        
