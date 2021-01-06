import cv2, os, numpy as np
from PIL import Image

wajahDir = 'datawajah'
latihDir = 'latihwajah'
def getImageLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    facesamples = []
    faceIDs = []
    for imagePath in imagePaths:
        PILImg = Image.open(imagePath).convert('L')
        imgNum = np.array(PILImg, 'uint8')
        faceID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgNum)
        for (x, y, w, h) in faces:
            facesamples.append(imgNum[y:y + h, x:x + w])
            faceIDs.append(faceID)
    return facesamples, faceIDs

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print("mesin sedang melakukan trainig data wajah, tunngu ya")
faces, IDs = getImageLabel(wajahDir)
faceRecognizer.train(faces, np.array(IDs))

# simpan
faceRecognizer.write(latihDir + '/latihan.xml')
print("sebanyak {0} data wajah telah di traning ke mesin.", format(len(np.unique(IDs))))
