import cv2, os, numpy as np

wajahDir ='datawajah'
latihDir ='latihwajah'
video = cv2.VideoCapture(0)
video.set(3, 640)
video.set(4, 400)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
eye = cv2.CascadeClassifier('eye-detect.xml')
faceRecognizer.read(latihDir+'/latihan.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['tidak tau','Rofii','Nama lain']


ninwidth = 0.1*video.get(3)
ninheight = 0.1*video.get(4)

while True:
    ratV, frame = video.read()
    frame = cv2.flip(frame, 1) #vertical flip
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(frame, 70, 70)
    faces = faceDetector.detectMultiScale(gray, 1.2, 5,minSize=(round(ninwidth),round(ninheight)),)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        id, confidence = faceRecognizer.predict(gray[y:y+h,x:x+w])
        if confidence<=50 :
            nameID = names[0]
            confidenceTxt = "{0}%".format(round(100-confidence))
        else:
            nameID = names[0]
            confidenceTxt = "{0}%".format(round(100 - confidence))

        cv2.putText(frame,str(nameID), (x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(frame, str(confidenceTxt), (x + 5, y + 200), font, 1,(255, 255, 0),1)
        roi_warna = frame[y:y + h, x:x + w]
        roi_gray = gray[y:y + h, x:x + w]
        mata = eye.detectMultiScale(roi_gray, 1.5, 3)
        for (mx, my, mw, mh) in mata:
            cv2.rectangle(roi_warna, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)

    cv2.imshow('Recognisi wajah and eye detect', frame)
    cv2.imshow('edge detect', edge)
    exit = cv2.waitKey(1) & 0xff
    if exit == ord('x'):
        break
print ("Exit")
video.release()
cv2.destroyAllWindows()
