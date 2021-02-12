import numpy as np
import pandas as pd
import cv2
import pickle
import matplotlib.pyplot as plt

import imutils
from shapely.geometry import Point, Polygon

import yolo3_one_file_to_detect_them_all as yolo3
import YOLOV3 as yolo3_custom
from keras.models import load_model

import progressbar
from lib import utils
from datetime import datetime
import argparse

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument('--weight', required=True, help='Path to the weights of pretrained YOLO model')
arg_parse.add_argument('--video_input', required=True, help='Video to be analyzed')
# arg_parse.add_argument('--court', required=True, help='Court image to calculate homography transform')
arg_parse.add_argument('--player1', required=True, help='Player 1 name')
arg_parse.add_argument('--player2', required=True, help='Player 2 name')
arg_parse.add_argument('--video_output', required=True, help='Output video save path')
arg_parse.add_argument('--player_info', required=True, help="Output players' coordinates save path")
args = vars(arg_parse.parse_args())

# initiate YOLOV3 model
model = yolo3_custom.make_yolov3_model()
weight_reader = yolo3_custom.WeightReader(args['weight'])
weight_reader.load_weights(model)

net_h, net_w = 416, 416 # input width & height
obj_thresh, nms_thresh = 0.5, 0.45 # threshold values
anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]] # player anchors
labels = ["person"] # YOLOV3 pre-trained labels
scaling_factor = 0.5 # scaling

# source points, a.k.a input court points
src_pts = np.array([
    [1, 86], 
    [1, 326], 
    [413, 326], 
    [413, 86]
])

# destination points, a.k.a output court points
dst_pts = np.array([
      [1,  1],     
      [1,  413],    
      [415,  415],    
      [415,  1],  
    ])

# analyze the input video
vid_path = args['video_input']
save_path = args['video_output']

cap = cv2.VideoCapture(vid_path)

scaling_factor = 0.5
video_FourCC = int(cap.get(cv2.CAP_PROP_FOURCC))
video_fps    = cap.get(cv2.CAP_PROP_FPS)
video_size   = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scaling_factor), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scaling_factor))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

writer = None
grabbed = True
frame_number = 0

# players dictionary
PLAYER_1 = args['player1']
PLAYER_2 = args['player2']
player_info = {'player_A': {'label': PLAYER_1, 'player_coords': [], 'player_torso': [], 'tracking_coords': []},
               'player_B': {'label': PLAYER_2, 'player_coords': [], 'player_torso': [], 'tracking_coords': []},
              }

# writer = cv2.VideoWriter(save_path, -1, video_fps, video_size) # set a video writer to save video output
bar = progressbar.ProgressBar(maxval=n_frames, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]) # progress bar for visualization purposes
bar.start()
start_time = datetime.now()

coordinates = []
while grabbed:
    frame_number += 1
    grabbed, frame = cap.read()
        
    if grabbed:
        frame = cv2.resize(frame, (video_size[0], video_size[1]), interpolation=cv2.INTER_AREA)

        # players' positions
        frame_h, frame_w, _ = frame.shape
        new_frame = yolo3_custom.preprocess_input(frame, net_h, net_w)
        
        preds = model.predict(new_frame)
        bboxes = []
        for i in range(len(preds)):
            bboxes += yolo3_custom.decode_netout(preds[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
        
        yolo3_custom.correct_yolo_boxes(bboxes, frame_h, frame_w, net_h, net_w)
        yolo3_custom.do_nms(bboxes, nms_thresh)
        frame = yolo3_custom.draw_boxes(frame, frame_number, bboxes, labels, obj_thresh, src_pts, player_info, player=player_info['player_A']['label'])

        # write each frame in a new video
#         writer.write(frame)
        
        cv2.imshow('Final Result', frame)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
    
        bar.update(frame_number) # updates the progress bar
    
    else:
        grabbed = False

print("Finished")
print("Elapsed time:", datetime.now() - start_time)
writer.release()
cap.release()
cv2.destroyAllWindows()

pickle.dump(player_info, open(args['player_info'], 'wb'))
