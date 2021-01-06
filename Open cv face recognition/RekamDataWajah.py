import cv2, os
wajahDir = 'datawajah'
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('eye-detect.xml')
video = cv2.VideoCapture(0)
faceID = input("masukan face ID yang akan di rekam datanya [tekan enter]: ")
print ("tatap wajah ke webcam, tunggu pengambilan data sampai selesai")
ambilData = 1
while True:
    ratV, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #edge = cv2.Canny(frame, 70, 70)
    muka = face.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in muka:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 4)
        namaFile = 'wajah.'+str(faceID)+'.'+str(ambilData)+'.jpg'
        cv2.imwrite(wajahDir+'/'+namaFile,frame)
        ambilData += 1
        roi_warna = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y + h, x:x + w]
        mata= eye.detectMultiScale(roi_gray, 1.5, 3)
        for (mx,my,mw,mh) in mata:
            cv2.rectangle(roi_warna,(mx, my), (mx+mw, my+mh), (255,255,0), 2)

    cv2.imshow('face and eye detect', frame)
    #cv2.imshow('edge detect', edge)
    exit = cv2.waitKey(1) & 0xff
    if exit == ord('x'):
        break
    elif ambilData>30:
        break
print ("pengambilan data selesai")
cv2.destroyAllWindows()
video.release()