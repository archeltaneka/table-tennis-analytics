import cv2

def init_court_capture(file_path, scaling_factor):
    
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    
    width = int(frame.shape[1] * scaling_factor)
    length = int(frame.shape[0] * scaling_factor)
    img = cv2.resize(frame, (width, length), interpolation=cv2.INTER_AREA)
    
    return img