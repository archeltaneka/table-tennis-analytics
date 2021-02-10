import numpy as np
import cv2
import argparse
from lib.utils import init_court_capture

# def left_click_event(event, x, y, flags, param):
    
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(img_copy, center=(x,y), radius=3, color=(0,255,0), thickness=-1)
#         src_pts.append((x,y))
        
#         if len(src_pts) == 1:
#             cv2.imshow('court', img_copy)
        
#         if len(src_pts) >= 2:
#             cv2.line(img_copy, pt1=src_pts[-1], pt2=src_pts[-2], color=(0,255,0), thickness=2)
#             cv2.imshow('court', img_copy)
            
#         if len(src_pts) == 4:
#             court = img_copy.copy()
#             cv2.polylines(court, [np.array(src_pts)], isClosed=True, color=(0,255,255), thickness=2)
            
#             cv2.imshow('court', court)

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument('--type', required=True, help='Input type ("video" or "picture")')
arg_parse.add_argument('--input', required=True, help='Image/video input path')
arg_parse.add_argument('--scaling', required=True, help='Scaling factor')
arg_parse.add_argument('--output', required=True, help='Save path')
args = vars(arg_parse.parse_args())

img = None
src_pts = []

if args['type'].lower() == 'video':
    img = init_court_capture(args['input'], float(args['scaling']))
elif args['type'].lower() == 'picture':
    img = cv2.imread(args['input'])
    
img_copy = img.copy()
    
cv2.imshow('court', img_copy)
# cv2.setMouseCallback('court', left_click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print('Court points:')
# print(src_pts)

# cv2.imwrite(args['output'], img)