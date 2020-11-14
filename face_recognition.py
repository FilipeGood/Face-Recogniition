import numpy as np
import cv2
import time
import os
import pandas as pd
import sys
from mss import mss
from PIL import Image


def test_video():
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    print('\n\nPress ESC to quit....')

    while(True):
        ret, frame = cap.read()
        #frame = cv2.flip(frame, -1) # Flip camera vertically
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', frame)
        cv2.imshow('gray', gray)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return



def read_ids():
    ids = pd.read_csv('ids.csv')
    return ids

def create_dataset():
    ids = read_ids()
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) # set video width
    cap.set(4, 480) # set video height

    font = cv2.FONT_HERSHEY_SIMPLEX

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    face_name = input('\n Enter user name and press <ENTER>: ')

    image_id = 0
    if face_name in ids['Name'].values:
        image_id = int(ids[ids['Name'] == face_name]['last_photo_id'])+1

    count = 0


    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            image_id +=1
            cv2.imshow('You :)', frame)

        text = 'Saved '+ str(image_id) +' images. Please continue to change the facial expressions :)'
        # cv2.putText(frame, text, (15, 15), font, 0.5, (255, 255, 255), 1)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        cv2.imwrite("dataset/"+str(face_name) + '.' +
                        str(image_id) + ".jpg", gray)
        cv2.imshow('You :)', frame)

        time.sleep(0.03)
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
            break

    if face_name in ids['Name'].values:
        # If it is a new name
        ids.loc[ids['Name'] == face_name,['last_photo_id'] ] = image_id
    else:
        if ids.shape[0] == 0:
            # if there is no ids
            current_id = 0
        else:
            last_id =  ids.tail(1)['ID']
            current_id = int(last_id)+1

        aux_df = pd.DataFrame([[current_id, face_name, image_id]], columns= ['ID', 'Name', 'last_photo_id'])
        ids = ids.append(aux_df)

    ids.to_csv('ids.csv', index=False)

    cap.release()
    cv2.destroyAllWindows()
    return ids

def get_training_data(folder_path = 'dataset'):
    ids_df = read_ids()

    images = os.listdir(folder_path)
    faces = []
    labels = []
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

    for img in images:
        with_face =  False
        name = img.split('.')[0]
        label = int(ids_df[ids_df.Name == name]['ID'])

        path_to_image = folder_path+'/'+img
        image = cv2.imread(path_to_image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = face_detector.detectMultiScale(image, 1.3, 5)
        for (x,y,w,h) in face:
            with_face = True

            faces.append(image[y:y+h,x:x+w])
            labels.append(label)
            cv2.destroyAllWindows()
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    return faces, labels

def train():
    faces, labels = get_training_data()
    print('Training {0} with images and {1} unique faces...'.format(len(labels), len(np.unique(np.array(labels)))))

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write('trainer.yml')


def recognize_webcam():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    faceCascade =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX

    ids_df = read_ids()
    names = ['None']+list(ids_df['Name'].values)+['Unkown']

    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height
    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            label, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            print(label, confidence)
            if (confidence > 20):
                label = names[label]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                label = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img,str(label),(x+5,y-5),font,1,(255,255,255),2)
            cv2.putText(img,str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

        cv2.imshow('camera',img)
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()



def recognize_screen():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    faceCascade =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX

    ids_df = read_ids()
    names = ['None']+list(ids_df['Name'].values)+['Unkown']

    #bounding_box = {'top': 100, 'left': 0, 'width': 400, 'height': 300}
    bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

    sct = mss()

    minW = 0.1*1920
    minH = 0.1*1080
    while True:
        sct_img = sct.grab(bounding_box)
        img = np.array(sct_img)
        #cv2.imshow('screen', np.array(sct_img))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            label, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            print(label, confidence)
            if (confidence > 20):
                label = names[label]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                label = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img,str(label),(x+5,y-5),font,1,(255,255,255),2)
            cv2.putText(img,str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

        cv2.imshow('camera',img)
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()



def main():
    if len(sys.argv)-1 == 0:
        print('Missing argument. Please add one of the following: test_video, create_dataset, train, predict_webcam or predict_screen')
        return None
    arg = sys.argv[1]

    if arg == 'test_video':
        test_video()
    elif arg == 'create_dataset':
        create_dataset()
    elif arg == 'train':
        train()
    elif arg == 'predict_webcam':
        recognize_webcam()
    elif arg == 'predict_screen':
        recognize_screen()
    else:
        print('Invalid argument! Please add one of the following: test_video, create_dataset, train or predict')

if __name__ == "__main__":
    """
        Three possibilities:
            - test_video
            - Take photos
            - Train
            - Predict
    """
    recognize_screen()
    #main()