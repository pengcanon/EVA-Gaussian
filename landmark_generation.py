import os
import json
import cv2
import mediapipe as mp
import warnings
from argparse import ArgumentParser
import subprocess
import face_recognition

import mmcv
from xtcocotools.coco import COCO

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result, process_mmdet_results)
from mmpose.datasets import DatasetInfo
from mmdet.apis import init_detector, inference_detector
import warnings
warnings.simplefilter("ignore")

# prepare models
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1)

def download_checkpoint(path, name, url):
	if os.path.isfile(path+name) == False:
		print("checkpoint(model) file does not exist, now download ...")
		subprocess.run(["wget", "-P", path, url])
 
device = 'cuda:0'
path = "./pretrained/"
checkpoint = "hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth"
url = "https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth"
download_checkpoint(path, checkpoint, url)

config = "./landmark_config.py"
model = init_pose_model(config, path+checkpoint, device)

dataset_path = 'path to your dataset'

for stage in ['train']:
    root_path = os.path.join(dataset_path, stage, 'img')
    results = {}
    for dirs in os.listdir(root_path):
        print(dirs)
        if dirs not in results:
            results[dirs] = {}

        for file in ['0.jpg', '1.jpg']
            if file not in results[dirs]:
                results[dirs][file] = {}

            if file.endswith('.jpg'):
                file_path = os.path.join(root_path, dirs, file)

                # face landmark detection
                data = face_recognition.load_image_file(file_path)
                face_det_results = face_recognition.face_locations(data)
            
                face_bbox_results = []
                
                for rect in face_det_results:
                    person = {}
                    person["bbox"] = [rect[3], rect[0], rect[1], rect[2]]
                    face_bbox_results.append(person)

                if 'face' not in results[dirs][file] :
                    results[dirs][file]['face']= {}

                pose_results, returned_outputs = inference_top_down_pose_model(model, file_path, face_bbox_results, bbox_thr=None, format='xyxy')
                for result in pose_results:
                    results[dirs][file]['face'] = result['keypoints'].tolist()

                # hand landmark detection
                frame = cv2.imread(file_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                hand_landmark_results = hands.process(frame)

                if hand_landmark_results.multi_hand_landmarks is not None:
                    for hand_index, hand_info in enumerate(hand_landmark_results.multi_handedness):
                        if len(hand_landmark_results.multi_handedness)==1:
                            hand_landmarks = hand_landmark_results.multi_hand_landmarks[0]
                        else:
                            hand_landmarks = hand_landmark_results.multi_hand_landmarks[hand_info.classification[0].index]
                        side = hand_info.classification[0].label
                        if side not in results[dirs][file] :
                            results[dirs][file][side]= {}

                        for i in range(21):
                            if i not in results[dirs][file][side] :
                                results[dirs][file][side][i]= {}
                                results[dirs][file][side][i]= {}

                            x = hand_landmarks.landmark[i].x*frame.shape[1]
                            y = hand_landmarks.landmark[i].y*frame.shape[0]

                            results[dirs][file][side][i]['x']=x
                            results[dirs][file][side][i]['y']=y

output_file = os.path.join(dataset_path, 'landmark.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)

print("Processing complete, data saved to: ", output_file)