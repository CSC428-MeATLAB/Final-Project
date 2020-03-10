import cv2 as cv
import numpy as np
from pynput.keyboard import Controller

# color ranges of targets
color_ranges = [
    [np.array([54,127,63]), np.array([64,255,198])],    # green
    [np.array([-4,127,77]), np.array([4,255,190])],     # red
    [np.array([26,127,77]), np.array([36,255,190])],    # yellow
    [np.array([102,127,77]), np.array([109,255,190])],  # blue
    [np.array([14,127,68]), np.array([20,255,198])]     # orange
]

class GuitarInfo:
    stddevThreshold = 10
    saturationThreshold = 100
    samples = 10
    tstep = 0.01
    keys = ['a', 's', 'j', 'k', 'l']

    def __init__(self):
        self.lines = None       # [x1, y1, x2, y2]
        self.targets = None     # [x, y] centroids of targets
        self.targetStats = None # [x, y, width, height, area] of targets
        self.detectedHolds = [False for _ in range(5)]
        self.lastDetected = [-100 for _ in range(5)]
        self.keysDown = [False for _ in range(5)]
        self.keyboard = Controller()
        self.count = 0
        self.initiated = False

    # takes a BGR image
    # extracts info about target locations and guitar string lines
    def extractInfo(self, im_bgr):
        dimensions = im_bgr.shape
        im_gray = cv.cvtColor(im_bgr, cv.COLOR_BGR2GRAY)
        im_hsv = cv.cvtColor(im_bgr, cv.COLOR_BGR2HSV)
        
        canny = cv.Canny(im_gray, 125, 150, None, 3)
        houghLines = cv.HoughLinesP(canny, 1, np.pi / 180, 60, None, 100, 10)

        # throw out horizontal lines
        isVertical = np.abs(houghLines[:,0:,0] - houghLines[:,0:,2]) < np.abs(houghLines[:,0:,1] - houghLines[:,0:,3])
        houghLines = houghLines[isVertical]

        self.lines = []
        self.targets = []
        self.targetStats = []
        line_start_y = 0
        line_end_y = dimensions[0]
        for color_range in color_ranges:
            target, stats = self._find_target(im_hsv, color_range[0], color_range[1])
            [x, y, w, h, a] = stats
            
            # find line with one end closest to target
            shifted_target = np.copy(target)
            shifted_target[1] -= h / 2
            line = self._find_closest_line(target, houghLines)

            # make sure lower point comes first
            if line[1] < line[3]:
                line = np.roll(line, 2)
            
            # start of line should be just above target when it's popped up
            line_start_y += y - h
            if line[3] < line_end_y:
                line_end_y = line[3]

            self.lines.append(line)
            self.targets.append(target)
            self.targetStats.append(stats)

        line_start_y = line_start_y / len(self.lines)

        # adjust lines to only cover string
        for line in self.lines:
            ys = line[1] - line[3]
            xs = line[0] - line[2]
            t = (line_start_y - line[1]) / ys
            line[0] = round(line[0] + xs * t)
            line[1] = round(line_start_y)

            t = (line_end_y - line[1]) / ys
            line[2] = round(line[0] + xs * t)
            line[3] = round(line_end_y)

        self.initiated = True

    def detectHolds(self, im_bgr):
        im_hsv = cv.cvtColor(im_bgr, cv.COLOR_BGR2HSV)

        for i in range(5):
            line = self.lines[i]
            samples = []
            for step in range(self.samples):
                t = step * self.tstep
                x = int(round(line[0] + t * (line[2] - line[0])))
                y = int(round(line[1] + t * (line[3] - line[1])))
                samples.append(im_hsv[y,x])
            samples = np.array(samples)
            std = np.average(np.std(samples, axis=0))
            avg = np.average(samples, axis=0)
            if std < self.stddevThreshold and avg[1] > self.saturationThreshold:
                self.lastDetected[i] = self.count
                self.detectedHolds[i] = True
            else:
                self.detectedHolds[i] = False

    def _find_target(self, img, hsv_low, hsv_high):
        if hsv_low[0] > hsv_high[0]:
            hsv_low_high = np.copy(hsv_low)
            hsv_low_high[1] = 180
            hsv_high_low = np.copy(hsv_high)
            hsv_high_low[0] = 0
            mask = cv.bitwise_or(
                cv.inRange(hsv, hsv_low, hsv_low_high),
                cv.inRange(hsv, hsv_high_low, hsv_high)
            )
        else:
            mask = cv.inRange(img, hsv_low, hsv_high)

        retval, labels, stats, centroids = cv.connectedComponentsWithStats(mask)
        
        # find label with most pixels
        best_label = 1
        for i in range(2, len(stats)):
            if stats[i][4] > stats[best_label][4]:
                best_label = i

        return centroids[best_label], stats[best_label]

    def _find_closest_line(self, pt, lines):
        dist2 = np.minimum(
            (lines[:,0] - pt[0])**2 + (lines[:,1] - pt[1])**2,
            (lines[:,2] - pt[0])**2 + (lines[:,3] - pt[1])**2
        )
        order = np.argsort(dist2)

        return lines[order[0]]

    def updateKeys(self):
        for i in range(5):
            if self.detectedHolds[i]:
                if not self.keysDown[i]:
                    self.keysDown[i] = True
                    self.keyboard.press(self.keys[i])
            else:
                if self.keysDown[i] and self.count - self.lastDetected[i] > 10:
                    self.keysDown[i] = False
                    self.keyboard.release(self.keys[i])

    # draw debug info on a frame
    def drawDebug(self, img):
        for i in range(5):
            line = self.lines[i]
            if self.detectedHolds[i]:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            cv.line(img, (line[0], line[1]), (line[2], line[3]), color, 3, cv.LINE_AA)

        for targetStat in self.targetStats:
            [x, y, w, h, a] = targetStat
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
