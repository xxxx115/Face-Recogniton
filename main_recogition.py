import sys
import dlib
import cv2
import cv2
import numpy as np
import pickle
import putText_chinese
from putText_chinese import cv2_add_chinese_text

CNN_DETECTOR_PATH = "model/mmod_human_face_detector.dat"
PREDICTOR_PATH="model/shape_predictor_5_face_landmarks.dat"
RECOGNIZER_PATH="model/dlib_face_recognition_resnet_model_v1.dat"
DATABASE_PATH="face.database.pkl"

class Recognizer:
    def __init__(self):
        self.detector=dlib.cnn_face_detection_model_v1(CNN_DETECTOR_PATH)
        self.sp=dlib.shape_predictor(PREDICTOR_PATH)
        self.facerec=dlib.face_recognition_model_v1(RECOGNIZER_PATH)
        with open(DATABASE_PATH,"rb") as f:
            self.data=pickle.load(f)
            self.known_encodings=self.data["encodings"]
            self.known_name=self.data["names"]
    def find_best_match(self,face_encoding):
        distances=np.linalg.norm(self.known_encodings-face_encoding,axis=1)
        best_match_index=np.argmin(distances)
        min_distance=distances[best_match_index]
        if min_distance<0.5:
            return self.known_name[best_match_index],min_distance
        else:
            return "未知",min_distance
    def process_image(self,image):
        scale_factor=1
        small_image=cv2.resize(image,(0,0),fx=scale_factor,fy=scale_factor)
        rbg_small_image=cv2.cvtColor(small_image,cv2.COLOR_BGR2RGB)
        dets=self.detector(rbg_small_image)
        results=[]
        for d in dets:
            detection=d.rect
            shape=self.sp(rbg_small_image,detection)
            face_encoding=np.array(self.facerec.compute_face_descriptor(rbg_small_image,shape))
            name,dist=self.find_best_match(face_encoding)
            results.append(name)
            left=int(detection.left()/scale_factor)
            top=int(detection.top()/scale_factor)
            right=int(detection.right()/scale_factor)
            bottom=int(detection.bottom()/scale_factor)
            color=(0,255,0) if name !="未知" else(0,0,255)
            cv2.rectangle(image,(left,top),(right,bottom),color,2)
            cv2.rectangle(image,(left,bottom-35),(right,bottom),color,cv2.FILLED)
            label=f"{name}({dist:.2f})"
            image=cv2_add_chinese_text(image,label,(left+6,bottom-30),(255,255,255),15)
        return image,results
def main():
    recognizer=Recognizer()
    video_capture=cv2.VideoCapture(0)
    while True:
        ret,image=video_capture.read()
        if not ret:
            break
        processed_image,names=recognizer.process_image(image)
        cv2.imshow("image",processed_image)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()