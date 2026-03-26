"""
禁止使用集体照、合照裁剪的人脸作为注册照；
禁止使用美颜、P 图、滤镜处理过的照片；
禁止使用翻拍屏幕 / 照片的二次拍摄图；
禁止使用人脸偏转超过 30° 的侧脸、低头 / 仰头照；
禁止使用遮挡五官的照片（口罩、墨镜、头发挡眼等）；
禁止使用模糊、拖影、满是噪点的低质量照片；
禁止单个人放入 10 张以上高度重复的照片。
"""

'''
统一用识别摄像头，在固定位置、固定光线、固定距离下，让每个人拍摄 3 张标准照（正脸、微左、微右），一次性完成注册，保证所有照片的一致性。
'''

import sys
import dlib
import cv2
import numpy as np
import os
import pickle
PREDICTOR_PATH = "model/shape_predictor_5_face_landmarks.dat"
RECOGNIZER_MODEL_PATH = "model/dlib_face_recognition_resnet_model_v1.dat"
DATASET_FOLDER="dataset"
script_dir = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR=os.path.join(script_dir,DATASET_FOLDER)
try:
    detector=dlib.get_frontal_face_detector()
    sp=dlib.shape_predictor(PREDICTOR_PATH)
    facerec=dlib.face_recognition_model_v1(RECOGNIZER_MODEL_PATH)
except Exception as e:
    sys.exit("Could not load face recognition model")
def get_face_encoding_from_image_path(img_path):
    img_array=np.fromfile(img_path,dtype=np.uint8)
    img=cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    img_rbg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    dets=detector(img_rbg,1)
    if len(dets)!=1:
        print(f"{img_path}检测到{len(dets)}张人脸")
        return None
    face=dets[0]
    shape=sp(img_rbg,face)
    face_descriptor=facerec.compute_face_descriptor(img_rbg,shape)
    return np.array(face_descriptor)
def aggregate_encodings(encodings,weights=None):
    encodings=np.array(encodings)
    return np.average(encodings,axis=0,weights=weights)
def build_database():
    known_encodings=[]
    known_names=[]
    for person_name in os.listdir(DATASET_DIR):
        person_dir=os.path.join(DATASET_DIR,person_name)
        if not os.path.isdir(person_dir):
            continue
        person_encodings=[]
        angle_weight={}
        for img_file in os.listdir(person_dir):
            img_path=os.path.join(person_dir,img_file)
            encoding=get_face_encoding_from_image_path(img_path)
            if encoding is not None:
                person_encodings.append(encoding)
                if "front" in img_file.lower():
                    angle_weight[img_path]=0.5
                if "left"  in img_file.lower():
                    angle_weight[img_path]=0.25
                if "right" in img_file.lower():
                    angle_weight[img_path]=0.25
        if person_encodings:
            weights=[angle_weight.get(os.path.join(person_dir,img_file),1/len(person_encodings)) for img_file in os.listdir(person_dir)
                     if get_face_encoding_from_image_path(os.path.join(person_dir,img_file)) is not None]
            final_weight=aggregate_encodings(person_encodings,weights)
            known_encodings.append(final_weight)
            known_names.append(person_name)
    data={"encodings":known_encodings,"names":known_names}
    save_path="face.database.pkl"
    with open(save_path,"wb") as f:
        pickle.dump(data,f)
    print(f"人脸数据库构建完成，共收录 {len(known_names)} 人")

if __name__=="__main__":
    build_database()