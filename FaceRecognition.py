# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:50:05 2020

@author: SACHUU
"""

import cv2
import numpy as np
import pickle
from PIL import Image
import os
import threading
import time
import requests
from datetime import datetime
import csv


class face_rec:
    def __init__(self):
        
        print("================================Face Recognition==============================")
        self.csv_folder = os.path.join(os.getcwd(),"CSV_files")

	
    def face_d(self):
        #self.count =0
        self.cwd = os.getcwd()
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        emplist = []
        g = 0
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainer_t.yml")
        self.labels = {"person_name": 1}
        with open("labels_t.pickel", 'rb') as f:
            og_labels = pickle.load(f)
            self.labels = {v:k for k,v in og_labels.items()}
            
        self.name = "No"
        self.cap = cv2.VideoCapture(0)
        while True:
            ret, img = self.cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
                
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                #recognize?
                id_, conf = self.recognizer.predict(roi_gray)
                if conf >= 60 and conf <= 100:
                    #print(id_)
                    #print(self.labels[id_])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    self.name = self.labels[id_]
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(img, self.name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                    #print(self.name)
                    #eyes = self.eye_cascade.detectMultiScale(roi_gray)
                    #for (ex,ey,ew,eh) in eyes:
                     #   cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0,255,0), 2)
            cv2.imshow('img',img)
            k = cv2.waitKey(30) & 0xff
            if k==27:
                ret, img = self.cap.read()
                break
            if self.name != "No":
                now = datetime.now()          
                csv_file_name = "{}.csv".format(now.strftime("%x").replace("/","-"))
                if not os.path.exists(os.path.join(self.csv_folder,csv_file_name)):
                    with open(os.path.join(self.csv_folder,csv_file_name), 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(["Name", "Time"])
						
                if self.name not in emplist:
                    now = datetime.now()          
                    emp_time = now.strftime("%X")
                    data_list = [self.name,emp_time]
                    emplist.append(self.name)
                    with open(os.path.join(self.csv_folder,csv_file_name), 'a+', newline='') as file:
                        csv_writer = csv.writer(file)
                        csv_writer.writerow(data_list)

        self.cap.release()
        cv2.destroyAllWindows()

        
f = face_rec()
f.face_d()