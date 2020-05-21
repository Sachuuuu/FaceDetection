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
        print("Meeting.....")
        self.people = []
        self.list_of_people = self.face_detect()
        self.store_csv(self.list_of_people)
	
    def face_detect(self):
        self.cwd = os.getcwd()
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        g = 0
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainner.yml")
        self.labels = {"person_name": 1}
        with open("labels.pickle", 'rb') as f:
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
                id_, conf = self.recognizer.predict(roi_gray)
                if conf >= 60 and conf <= 100:

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    self.name = self.labels[id_]
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(img, self.name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                    if self.name not in self.people:
                        self.people.append(self.name)
                        

            cv2.imshow('img',img)
            k = cv2.waitKey(30) & 0xff
            if k==27:
                ret, img = self.cap.read()
                break

        self.cap.release()
        cv2.destroyAllWindows()
        return self.people
    
    def store_csv(self, data_list):
        current = os.getcwd()
        path = os.path.join(current,"Meeting_CSV")
        now = datetime.now()
        meeting_date = now.strftime("%x")
        meeting_time = now.strftime("%X")
        
        datetime_value = "{}_{}".format(meeting_date,meeting_time)
        csv_file_name = "Meetings.csv"
        if not os.path.exists(os.path.join(path,csv_file_name)):
            with open(os.path.join(path,csv_file_name), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Date and Time", "Names"])

        data_list = [datetime_value,data_list]
        with open(os.path.join(path,csv_file_name), 'a+', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(data_list)
        
f = face_rec()
