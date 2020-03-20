import cv2
import numpy as np
import time
from guitar_info import GuitarInfo
import sys
import mss
import time

if len(sys.argv) < 2:
    MODE = 0
else:
    MODE = 1
FRAME_SKIP_COUNT = 1
WINDOW_NUM = 2

WINDOW_TITLE = 'Video'
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_TITLE, 1500,1000)

guitar = GuitarInfo()

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

prevTime = time.time()
# Read until video is completed
while(isVideoOpen()):
    # Capture frame-by-frame
    ret, frame = readFrame()
    frame = cv2.resize(frame,
        (1280,720),
        0, 
        0, 
        interpolation=cv2.INTER_NEAREST)
    if ret == True:
        guitar.count += 1
        # Press Q on keyboard to  exit
        if not guitar.initiated:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('l'):
                guitar.extractInfo(frame)
                print("Found info")

        if guitar.initiated:
            guitar.detectHolds(frame)
            guitar.detectNotes(frame)
            guitar.updateKeys()
        #guitar.drawDebug(frame)
        #cv2.imwrite(f'render/{guitar.count:05}.png', frame)
        # Display the resulting frame
        if not guitar.initiated:
            cv2.imshow(WINDOW_TITLE,frame)

        if guitar.count % 60 == 0:
            curTime = time.time()
            print(60 / (curTime - prevTime))
            prevTime = curTime
    # Break the loop
    else:
        break

# When everything done, release the video capture object
if MODE == 1:
    cap.release()

# Closes all the frames
cv2.destroyAllWindows()
