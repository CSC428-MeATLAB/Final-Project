import cv2
import numpy as np
import time

# Minimim area threshold that is boxed
AREA_MIN_THRESHHOLD = 1000
AREA_MAX_THRESHOLD = 3000

# Number of frames to skip to calculate the box
FRAME_SKIP_COUNT = 1

# Control the framerate
DELAY = 1/20

# Title of the window
WINDOW_TITLE = 'Video'

# Percentage of screen to use for note detection
CROP_WIDTH = .5
CROP_HEIGHT = .5

# Define Window settings
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_TITLE, 1500,1000)

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('testvid.mp4')

# Load in a blue note template
template = cv2.imread("notecenter.png", cv2.IMREAD_GRAYSCALE)
temp_w, temp_h = template.shape[::-1]

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Function that takes in a image and draws boxes around suspected plants
def box_image(img: np.array):
    """
    # Converting image from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mask for blue notes
    mask1 = cv2.inRange(hsv, (100, 20, 20), (150, 255, 255))

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
        if AREA_MAX_THRESHOLD > cv2.contourArea(c) > AREA_MIN_THRESHHOLD:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # check the rectangle is note shaped
            if (4 > w/h > 2):
                print ("width:" + str(w) + "height:" + str(h))
                rect_list.append((x, y, w, h))
            # draw a green rectangle to visualize the bounding rect
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 15)
    """
    rect_list = []

    # Compute image bounds for template matching
    top_bound = int(img.shape[0] * (1-CROP_HEIGHT))
    bottom_bound = int(img.shape[0] * 14.0/15)
    left_bound = int(img.shape[1]/2 - img.shape[1] * (1-CROP_WIDTH)/2)
    right_bound = int(img.shape[1]/2 + img.shape[1] * (1-CROP_WIDTH)/2)

    # Crop image and template match it
    crop_img = img[top_bound:bottom_bound , left_bound:right_bound]
    gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= 0.7)

    # Add all the points to an array
    for pt in zip(*loc[::-1]):
        subimg = gray_img[pt[1] + int(.33 * temp_h) : pt[1] + int(.66 * temp_h) , pt[0] + int(.33 * temp_w) : pt[0] + int(.66 * temp_w)]
        center_shade = np.average(np.average(subimg, axis=0), axis=0) 
        if (center_shade > 200):
            rect_list.append((pt[0] + left_bound, pt[1] + top_bound, temp_w, temp_h))

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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        # Display the resulting frame
        cv2.imshow(WINDOW_TITLE,frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) == ord('q'):
            break

        # Slomo mode for testing
        if cv2.waitKey(1) == ord('f'):
            DELAY = 1/60
        if cv2.waitKey(1) == ord('s'):
           DELAY = 1/20

        # print(f'Frame {count} Calc Time: {t1-t0}')

    # Break the loop
    else:
        break

    time.sleep(DELAY)


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
