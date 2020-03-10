import cv2
import numpy as np
import time
from guitar_info import GuitarInfo
import sys
import mss

FRAME_SKIP_COUNT = 2
MODE = 0
WINDOW_NUM = 1

WINDOW_TITLE = 'Video'
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_TITLE, 1500,1000)

# current frame counter
count = 0
guitarInfo = None

if MODE == 0:
    sct = mss.mss()
    mon = sct.monitors[WINDOW_NUM]
    readFrame = lambda: (True, np.array(sct.grab(mon)))
    isVideoOpen = lambda: True
elif MODE == 1:
    cap = cv2.VideoCapture(sys.argv[1])
    isVideoOpen = lambda: cap.isOpened()
    readFrame = lambda: cap.read()
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

# Read until video is completed
while(isVideoOpen()):
    # Capture frame-by-frame
    ret, frame = readFrame()
    if ret == True:
        count = count + 1            

        if ((count % FRAME_SKIP_COUNT) == 0):
            if guitarInfo is not None:
                guitarInfo.detectHolds(frame)

        if guitarInfo is not None:
            guitarInfo.drawDebug(frame)

        # Display the resulting frame
        cv2.imshow(WINDOW_TITLE,frame)

        # Press Q on keyboard to  exit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('l'):
            if guitarInfo is None:
                guitarInfo = GuitarInfo()
            guitarInfo.extractInfo(frame)
            print("Found info")

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
