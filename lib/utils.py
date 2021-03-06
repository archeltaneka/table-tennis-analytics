import numpy as np
import cv2
import matplotlib.pyplot as plt

def init_court_capture(file_path, scaling_factor):
    
    """
    Captures court image in a video file
    
    Arguments:
        file_path: String - Absolute path to the video file
        scaling_factor: Float - Scales up/down the resulting image
        
    Outputs:
        img: numpy.ndarray - Captured court image
    """
    
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    
    width = int(frame.shape[1] * scaling_factor)
    length = int(frame.shape[0] * scaling_factor)
    img = cv2.resize(frame, (width, length), interpolation=cv2.INTER_AREA)
    
    return img

def drawPlayers(im, pred_boxes, pred_classes, showResult=False):
    
    """
    Draws circles on the detected players
    
    Arguments:
        im: numpy.ndarray - Captured court image
        pred_boxes: List - Players' bounding boxes
        pred_classes: List - List of labels (we only use "Person" in this case)
        showResult: Boolean - Show the circles plotted on the image
        
    Outputs:
        None
    """
    
    # box config
    color = [255, 0, 0]
    thickness = 3
    radius = 3

    i  = 0
    for box in pred_boxes:

        # get box's x & y positions
        x1 = int(box.xmin)
        y1 = int(box.ymin)
        x2 = int(box.xmax)
        y2 = int(box.ymax)

        xc = x1 + int((x2 - x1)/2) # center point
        player_pos = (xc - 1, y2 - 25) # player's position

        court = Polygon(src_pts)

        # Draw only players that are within the court
        if (box.classes[0] > obj_thresh) & (Point(player_pos).within(court)):
            if showResult:
                print("[% 3d, % 3d]" %(xc, y2))

            cv2.circle(im, player_pos, radius, color, thickness)
            i = i + 1            

    if showResult:
        plt.imshow(im)
#         cv2.imshow('Court', im)
        
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

def homographyTransform(img_src, img_dst, showResult=False):
    
    """
    Performs homography transformation between a captured frame of a table tennis game and a table tennis court
    
    Arguments:
        img_src: numpy.ndarray - Captured frame of a table tennis game
        img_dst: numpy.ndarray - Court image
        showResult: Boolean - Show the circles plotted on the image
        
    Outputs:
        img_out: numpy.ndarray - The transformed image
    """
    
    # Calculate Homography
    h, status = cv2.findHomography(src_pts, dst_pts)
    img_out = cv2.warpPerspective(img_src, h, (img_dst.shape[1], img_dst.shape[0]))
    
    if showResult:
        plt.imshow(img_out)
#         cv2.imshow('Court', img_out)
        
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

    return img_out

def getPlayersMask(im):
    
    """
    Performs player masking to focus only on the players and ignores everything else
    
    Arguments:
        im: numpy.ndarray - The resulted image from homography transformation
    
    Outputs:
        numpy.ndarray - The result of masked image
        
    """
    
    lower_range = np.array([255,0,0])                         # Set the Lower range value of blue in BGR
    upper_range = np.array([255,155,155])                     # Set the Upper range value of blue in BGR
    mask = cv2.inRange(im, lower_range, upper_range)          # Create a mask with range
    result = cv2.bitwise_and(im, im, mask = mask)             # Performing bitwise and operation with mask in img variable
    # cv2_imshow(result)                              

    return cv2.inRange(result, lower_range, upper_range) 

def drawPlayersOnCourt(im, coord, color, radius=10):
    
    """
    Draw circles on players in the original court image
    
    Arguments:
        im: numpy.ndarray - Court image
        coord: numpy.ndarray - Circle coordinates (X,Y)
        color: List - RGB color values
        radius: Integer - Circle radius
        
    Outputs:
        im: numpy.ndarray - The resulted image after adding circles
    """
    
    # draw a circle on players' positions according to their coordinates
    for player_pos in coord:
        center = (player_pos[0], player_pos[1])
        cv2.circle(im, center, radius, color, thickness=-1)
        
    return im

def generate_heatmap(court_path, coords, player, video_duration, bins=25, save_path):
    
    """
    Plot heatmap figure according to players' coordinates
    
    Arguments:
        court_path: String - Path to the court image
        coords: numpy.ndarray - Player coordinates
        player: String - Player name
        video_duration: Float - Video duration
        bins: Integer: Number of bins to show
        save_path: String - Path to save the heatmap figure
        
    Outputs:
        None
    """
    
    pixel_to_meter = 0.0002645833
    diff = [(x1[0]-x0[0], x1[1]-x0[1]) for x0,x1 in zip(coords[0::], coords[1::])]
    dist = [np.sqrt(x**2 + y**2) for x,y in diff]
    travel_distance = int(np.sum(dist) * pixel_to_meter)
    speed = np.round((travel_distance/video_duration) * 3600/1000, 1)
    
    court = cv2.imread(court_path)
    
    pos_x = [pos[0] for pos in coords]
    pos_y = [pos[1] for pos in coords]
    hm, edge_x, edge_y = np.histogram2d(pos_x, pos_y, bins=bins, normed=True)
    extent = [0, court.shape[1], court.shape[0], 1]
    
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.set_xlim(0, court.shape[1])
    ax.set_ylim(court.shape[0], 0)
    
    fig1 = ax.imshow(court[:,:,[2,1,0]])
    fig2 = ax.imshow(hm.T, cmap='hot_r', alpha=0.8, interpolation='gaussian', extent=extent)
    
    plt.axis('off')
    plt.title('Heatmap: {} (1st Set)'.format(player))
    plt.annotate('Court Coverage: {}m\n'.format(travel_distance), (.19,.1), xycoords='figure fraction')
    plt.annotate('Speed: {}km/h'.format(speed), (.19,.1), xycoords='figure fraction')
    plt.show()
    
    if save_path:
        plt.savefig(save_path)