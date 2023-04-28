#import all necessary pakages
import cv2

#Deepface is a library use for easy detection of face atribute and its analysis
from deepface import DeepFace
from cv2 import CascadeClassifier

classifier=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)  #cap object captures the vedio

#while loop will continously captures the vedio
while True:

    rec,frame=cap.read()  #captures the vedio in form of frames
    frame=cv2.cvtColor(frame,0)
    
    detections=classifier.detectMultiScale(frame,1.3,5) #detect the faces from training set from classifier
    response=DeepFace.analyze(frame,actions=("emotion",),enforce_detection=False)

    #store the all emotion prediction with there percentage of accuracy
    emotionPredPer=response[0]
    
    #emotionPerdPer is an dictionary extract the dominant value of emotion
    dominantEmotion=emotionPredPer['dominant_emotion']
    
    cv2.putText(frame,text="press 'q' to quit.",org=(20,20),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(0,0,255))

    if(len(detections)>0):

        (x,y,w,h)=detections[0]
        cv2.putText(frame,text=dominantEmotion,org=(x,y),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(0,0,255))
        new_frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
