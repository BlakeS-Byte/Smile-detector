#Detect and crop face from the designated image, then
#convert it to greyscale and rescale down to 48x48 pixels
import cv2
import glob
import shutil
import os
from PIL import Image

# deletes all previously formatted photos
direct = 'custom/formatted/'
for f in os.listdir(direct):
    os.remove(os.path.join(direct, f))

# copies all of the photos into /formatted to be processed
for filename in glob.glob('custom/*.jpg'):
    shutil.copy(filename, 'custom/formatted/')

# goes through all custom images and formats to be worked on
for filename in glob.glob('custom/formatted/*.jpg'):

    #Read the input image
    img = cv2.imread(filename)

    #Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Load haar cascade
    detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    #Detect faces
    faces = detector.detectMultiScale(gray, 1.1, 5)

    #Draw rectangle around face and crop face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        cv2.imwrite(filename, faces)

    #Convert cropped image to grayscale and scale down to 48x48 pixels
    scaling = Image.open(filename).convert('L')
    scaled_face = scaling.resize((48, 48))
    scaled_face.save(filename, 'PNG', optimize=True)
