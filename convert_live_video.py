import cv2
import numpy as np
import time

# Minimim area threshold that is boxed
AREA_THRESHHOLD = 1000

# Number of frames to skip to calculate the box
FRAME_SKIP_COUNT = 2

# Title of the window
WINDOW_TITLE = 'Video'

# Define Window settings
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_TITLE, 1500,1000)

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Function that takes in a image and draws boxes around suspected plants
def box_image(img: np.array):
    # Converting image from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generating the mask that outlines the plants
    # Method 1: Look for the color green
    mask1 = cv2.inRange(hsv, (30, 30, 30), (70, 255,255))
    # Method 2

    # Take the mask and clean up the holes in the mask
    # Open removes area of the holes in the mask (removes noise) and then adds area to the holes
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    # Dilate areas in the mask (Add area to the holes in the mask)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    ret,thresh = cv2.threshold(mask1, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # List of Rectangle objects
    rect_list = []
    # Loop through each of the "Plant" areas
    for c in contours:
        # if the "Plant" is large enough draw a rectangle around it
        if cv2.contourArea(c) > AREA_THRESHHOLD:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            rect_list.append((x, y, w, h))
            # draw a green rectangle to visualize the bounding rect
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 15)
    return rect_list

# current frame counter
count = 0
rect_list = []

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # t0 = time.time()
        count = count + 1
        if ((count % FRAME_SKIP_COUNT) == 0):
            rect_list = box_image(frame)
        # t1 = time.time()

        for rects in rect_list:
            x, y, w, h = rects
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 15)

        # Display the resulting frame
        cv2.imshow(WINDOW_TITLE,frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) == ord('q'):
            break

        # print(f'Frame {count} Calc Time: {t1-t0}')

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
