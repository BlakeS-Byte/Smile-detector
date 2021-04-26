#Detect and crop face from the designated image, then
#convert it to greyscale and rescale down to 48x48 pixels
import cv2
from PIL import Image


#Read the input image
img = cv2.imread('20210425_164112.jpg')

#Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Load haar cascade
detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

#Detect faces
faces = detector.detectMultiScale(gray, 1.1, 4)

#Draw rectangle around face and crop face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), ( x +w, y+ h), (0, 0, 255), 2)
    faces = img[y:y + h, x:x + w]
    #cv2.imshow("face", faces)
    cv2.imwrite('face.png', faces)

#Saves copy of original image with detection box around face
cv2.imwrite('detection.jpg', img)

#Convert cropped image to grayscale and scale down to 48x48 pixels
scaling = Image.open('face.png').convert('LA')
scaled_face = scaling.resize((48,48))
scaled_face.save('face.png', 'PNG', optimize=True)

