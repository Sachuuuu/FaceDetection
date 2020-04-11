import cv2
import numpy as np
import pickle
from PIL import Image
import os



class face_train:
    def __init__(self):

        
        self.cwd = os.getcwd()
        
        self.face_cascade = cv2.CascadeClassifier(os.path.join(self.cwd,'haarcascade_frontalface_default.xml'))
        self.image_dir = os.path.join(self.cwd,'images')
        
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.current_id = 0
        self.label_ids = {}
        self.y_labels = []
        self.x_train = []
        print( self.image_dir)
        self.face_d()
    
    def face_d(self):
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith("jpg") or file.endswith("jpeg"):
                    path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(path))
                    #print(label, path)
                    if not label in self.label_ids:
                        self.label_ids[label] = self.current_id
                        self.current_id += 1
                        id_ = self.label_ids[label]
                        #print(self.label_ids)
                        #self.y_labels.append(label) #some number
                        #self.x_train.append(path) #verify this image,turn into a numpy array,gray
                        pil_image = Image.open(path).convert("L")# grayscale
                        size = (550, 550)
                        final_image = pil_image.resize(size, Image.ANTIALIAS)
                        image_array = np.array(pil_image, "uint8")
                        #print(image_array)
                        faces = self.face_cascade.detectMultiScale(image_array, 1.3, 5)
                        for (x,y,w,h) in faces:
                            roi = image_array[y:y+h, x:x+w]
                            self.x_train.append(roi)
                            self.y_labels.append(id_)
		#print(self.y_labels)
		#print(self.x_train)
        
    
        with open('labels2.pickel', 'wb') as f:
            pickle.dump(self.label_ids, f)

        self.recognizer.train(self.x_train, np.array(self.y_labels))
        self.recognizer.save('trainer2.yml')
        
d = face_train()