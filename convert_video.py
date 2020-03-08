import cv2
import numpy as np
import time

# Minimim area threshold that is boxed
AREA_MIN_THRESHHOLD = 1000
AREA_MAX_THRESHOLD = 3000

# Number of frames to skip to calculate the box
FRAME_SKIP_COUNT = 2

# Control the framerate
DELAY = 1/600

# Title of the window
WINDOW_TITLE = 'Video'

# Percentage of screen to use for note detection
CROP_WIDTH = .5
CROP_HEIGHT = .35

# Define Window settings
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_TITLE, 1500,1000)

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('bot-no-effects-720p.mp4')

# Load in a template
template = cv2.imread("notecenter.png", cv2.IMREAD_GRAYSCALE)
reg_temp_w, reg_temp_h = template.shape[::-1]

# Load in a hammer note template
ham_template = cv2.imread("hammercenter.png", cv2.IMREAD_GRAYSCALE)
ham_temp_w, ham_temp_h = ham_template.shape[::-1]

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Do non max supp using 2n+1*2n+1 region
def nonMaxSupp(pts, img):
    n = 2
    newx = []
    newy = []
    # for each pixel
    for pt in zip(*pts[::-1]):
        x = pt[0]
        y = pt[1]
 
        add = True
        # for a 2n+1 * 2n+1 window
        for i in range(x - n, x + n):
            for j in range(y - n, y + n):
                if ( n<x<img.shape[1]-n and n<y<img.shape[0]-n and img[y][x] < img[j][i]):
                    add = False
        if add:
            newx.append(x)
            newy.append(y)
    
    return np.array([newy,newx])


# Function that takes in a image and draws boxes around suspected plants
def box_image(img: np.array):

    rect_list = []

    # Compute image bounds for template matching
    top_bound = int(img.shape[0] * (1-CROP_HEIGHT))
    bottom_bound = int(img.shape[0] * 13.0/15)
    left_bound = int(img.shape[1]/2 - img.shape[1] * (1-CROP_WIDTH)/2)
    right_bound = int(img.shape[1]/2 + img.shape[1] * (1-CROP_WIDTH)/2)

    # Crop image
    crop_img = img[top_bound:bottom_bound , left_bound:right_bound]
    gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # Detect regular and hammer on notes
    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    result2 =cv2.matchTemplate(gray_img, ham_template, cv2.TM_CCOEFF_NORMED)

    # Filter down to notes we're sure about
    loc = np.where(result >= 0.8)
    loc = nonMaxSupp(loc, result)
    loc2 = np.where(result2 >= 0.7)
    loc2 = nonMaxSupp(loc2, result2)

    # Add regular notes to the array
    for pt in zip(*loc[::-1]):
        subimg = gray_img[pt[1] + int(.33 * reg_temp_h) : pt[1] + int(.66 * reg_temp_h) , pt[0] + int(.33 * reg_temp_w) : pt[0] + int(.66 * reg_temp_w)]
        center_shade = np.average(np.average(subimg, axis=0), axis=0)
        # Make sure the center of the note is white (so we dont detect the bottom bar) 
        if (center_shade > 190):
            rect_list.append((pt[0] + left_bound, pt[1] + top_bound, reg_temp_w, reg_temp_h))

    # Add hammer on notes to the array
    for pt in zip(*loc2[::-1]):
        subimg = gray_img[pt[1] + int(.33 * ham_temp_h) : pt[1] + int(.66 * ham_temp_h) , pt[0] + int(.33 * ham_temp_w) : pt[0] + int(.66 * ham_temp_w)]
        center_shade = np.average(np.average(subimg, axis=0), axis=0)
        # Make sure the center of the note is white (so we dont detect the bottom bar) 
        if (center_shade > 190):
            rect_list.append((pt[0] + left_bound, pt[1] + top_bound, ham_temp_w, ham_temp_h))

    return rect_list

# current frame counter
count = 0
rect_list = []

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.resize(frame,
                (1280,720),
                0, 
                0, 
                interpolation=cv2.INTER_NEAREST)

    if ret == True:
        # t0 = time.time()
        count = count + 1
        if ((count % FRAME_SKIP_COUNT) == 0):
            rect_list = box_image(frame)
        # t1 = time.time()

        for rects in rect_list:
            x, y, w, h = rects
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

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
