import numpy as np
import pandas as pd
import os
import cv2
import dlib
from math import hypot
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import euclidean
os.chdir('C:/Users/mike/Desktop/face_rec/own')

class Detector(object):
    def __init__(self, path = None):
        try:
            self.model = ('haar', cv2.CascadeClassifier(path))
        except:
            self.model = ('dlib', dlib.get_frontal_face_detector())
            #self.model = ('haar', cv2.CascadeClassifier('./detector/haarcascade_frontalface_default.xml'))
            
    def __call__(self, img):
        if self.model[0] == 'haar':
            return self.model[1].detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        elif self.model[0] == 'dlib':
            return [(f.left(),f.top(),f.width(),f.height()) for f in self.model[1](img)]

class Recognizer(object):
    def __init__(self, path = None, known = './known/faces.hdf5', use = 'default'):
        try:
            self.model = ('dlib', dlib.face_recognition_model_v1(path[0]), dlib.shape_predictor(path[1]))
        except:
            self.model = ('dlib', dlib.face_recognition_model_v1('./detector/dlib_face_recognition_resnet_model_v1.dat'),
                          dlib.shape_predictor('./detector/shape_predictor_5_face_landmarks.dat'))
        self.known = pd.HDFStore(known)
        self.use = use
        self.data = self.known.get(self.use)
    
    def __call__(self, img):
        if self.model[0] == 'dlib':
            return np.array(self.model[1].compute_face_descriptor(img, self.model[2](img,dlib.rectangle(0,0,img.shape[1],img.shape[0]))))
        
    def webcam_hash(self, name):
        cam = cv2.VideoCapture(0)
        stat, img = cam.read()
        self.add_hash(img=img,names=name)
    
    def add_hash(self, files = None, img = None, names = None):
        if files is not None:
            for i, file in enumerate(files):
                img = cv2.imread(file)
                det = Detector()
                ff = det(img)
                if len(ff) > 0:
                    (x,y,w,h) = ff[0]
                    img = img[y:y+h,x:x+w]
                    if names is None:
                        iname = os.path.splitext(os.path.basename(file))[0]
                    else:
                        iname = names[i]
                    ihash = self.__call__(img)
                    print('Adding hash for: {0}'.format(iname))
                    tmp = pd.DataFrame(ihash.reshape((1,128)), index = [iname])
                    if '/'+self.use in self.known.keys():
                        self.known.append(self.use, tmp, 't')
                    else:
                        self.known.put(self.use, tmp, 't')
        if img is not None:
            iname = names
            ihash = self.__call__(img)
            tmp = pd.DataFrame(ihash.reshape((1,128)), index = [iname])
            
    def compare_hash(self, img, thresh = 0.5):
        ihash = self.__call__(img)
        out = self.data.apply(lambda x: euclidean(x,ihash), 1)
        if out.min() < thresh:
            return out.idxmin()
        else:
            return 'Unknown'

class Emotion(object):
    def __init__(self, path = './emotion/weights'):
        try:
            self.model = None
        except:
            self.model = None
        
    def __call__(self, img):
        if self.model is None:
            return "happy"

class Face(object):
    def __init__(self, pos, name = 'Unknown', emo = 'neutral', col = (255,0,0)):
        self.lframe = 0
        self.name = name
        self.pos = pos
        self.vec = (0,0)
        self.emo = emo
        self.col = col
        self.visible = True
        self.outside = False
    
    def align(self, faces, thresh = 20):
        for face in faces:
            d = hypot(self.pos[0]-face[0],self.pos[1]-face[1])
            if d <= thresh:
                self.update_vec((face[0],face[1]))
                self.update_pos()
                break
    
    def step(self):
        self.pos = (self.pos[0] + self.vec[0], self.pos[1] + self.vec[1],
                    self.pos[2], self.pos[3])

def align_faces(faces, positions, thresh = 50):
    return [Face(face) for face in positions]

def get_names(img, detector, recognizer):
    out = []
    faces = detector(img)
    for (x,y,w,h) in faces:
        name = recognizer.compare_hash(img[y:y+h,x:x+w])
        out += [Face((x,y,w,h), name=name)]
    return out

class Camera(object):
    def __init__(self, detector = None, recognizer = None, emotion = None):
        self.cam = None
        self.detector = Detector(detector)
        self.recognizer = Recognizer(recognizer)
        self.faces = []
        self.frame = 0
        
    def run(self, update_steps=10):
        self.cam = cv2.VideoCapture(0)
        self.frame = 0
        while (True):
            img = self.get_frame()
            executor = ThreadPoolExecutor(max_workers=2)
            if self.frame == 0:
                f = executor.submit(get_names, img=img,detector=self.detector,recognizer=self.recognizer)
            if not f.running() and self.frame % update_steps == 0:
                self.faces = f.result()
                f = executor.submit(get_names, img=img,detector=self.detector,recognizer=self.recognizer)
            if len(self.faces) > 0:
                img = self.draw_faces(img, self.faces)
            cv2.imshow('frame',img)
            self.frame += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cam.release()
        cv2.destroyAllWindows()
        
    def get_frame(self):
        stat, img = self.cam.read()
        if stat:
            return img
        else:
            return self.img
    
    @staticmethod
    def draw_faces(img, faces):
        for face in faces:
            (x,y,w,h) = face.pos
            cv2.rectangle(img,(x,y),(x+w,y+h),face.col,2)
            tw,th = cv2.getTextSize(face.name,cv2.FONT_HERSHEY_COMPLEX,1,2)[0]
            cv2.rectangle(img,(x,y-th),(x+tw,y),face.col,-1)
            cv2.putText(img,face.name,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),bottomLeftOrigin=False)
        return img

def take_picture():
    cam = cv2.VideoCapture(0)
    stat, img = cam.read()
    cam.release()
    return img

a=Camera()
a.run(1)
