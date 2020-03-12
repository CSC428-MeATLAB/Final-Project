import cv2 as cv
import numpy as np
from pynput.keyboard import Controller
import math

# color ranges of targets
color_ranges = [
    [np.array([54,127,63]), np.array([64,255,198])],    # green
    [np.array([-4,127,77]), np.array([4,255,190])],     # red
    [np.array([26,127,77]), np.array([36,255,190])],    # yellow
    [np.array([102,127,77]), np.array([109,255,190])],  # blue
    [np.array([14,127,68]), np.array([20,255,198])]     # orange
]

note_colors = [
    (0, 255, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (0, 127, 255)
]

CROP_WIDTH = .4
CROP_HEIGHT = .35

# Load in a template
template = cv.imread("notecenter.png", cv.IMREAD_GRAYSCALE)
reg_temp_w, reg_temp_h = template.shape[::-1]

# Load in a hammer note template
ham_template = cv.imread("hammercenter.png", cv.IMREAD_GRAYSCALE)
ham_temp_w, ham_temp_h = ham_template.shape[::-1]

class GuitarInfo:
    stddevThreshold = 10
    saturationThreshold = 100
    samples = 10
    tstep = 0.01
    keys = ['a', 's', 'j', 'k', 'l']

    def __init__(self):
        self.lines = []       # [x1, y1, x2, y2]
        self.targets = []     # [x, y] centroids of targets
        self.targetStats = [] # [x, y, width, height, area] of targets
        self.detectedHolds = [False for _ in range(5)]
        self.lastDetected = [-100 for _ in range(5)]
        self.keysDown = [False for _ in range(5)]
        self.keyboard = Controller()
        self.count = 0
        self.initiated = False
        self.detectedNotes = [[] for _ in range(5)]

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

        if len(self.lines) != 5:
            print("Error: found {} lines".format(len(self.lines)))
            return

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

    def detectNotes(self, img):
        rect_list = []
        # Compute image bounds for template matching
        top_bound = int(img.shape[0] * (1-CROP_HEIGHT))
        bottom_bound = int(img.shape[0] * 13.0/15)
        left_bound = int(img.shape[1]/2 - img.shape[1] * (1-CROP_WIDTH)/2)
        right_bound = int(img.shape[1]/2 + img.shape[1] * (1-CROP_WIDTH)/2)

        # Crop image
        crop_img = img[top_bound:bottom_bound , left_bound:right_bound]
        gray_img = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)

        # Detect regular and hammer on notes
        result = cv.matchTemplate(gray_img, template, cv.TM_CCOEFF_NORMED)
        result2 = cv.matchTemplate(gray_img, ham_template, cv.TM_CCOEFF_NORMED)

        # Filter down to notes we're sure about
        loc = np.where(result >= 0.8)
        loc = self._nonMaxSupp(loc, result)
        loc2 = np.where(result2 >= 0.7)
        loc2 = self._nonMaxSupp(loc2, result2)

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
        
        # Determine which lane each note is in
        self.detectedNotes = [[] for _ in range(5)]
        for rect in rect_list:
            x, y, w, h = rect
            center = np.array([x + w / 2, y + h / 2])
            closestLine = None
            closestDist = math.inf
            for i in range(len(self.lines)):
                line = self.lines[i]
                p1 = np.array([line[0], line[1]])
                p2 = np.array([line[2], line[3]])
                d = np.abs(np.linalg.norm(np.cross(p2 - p1, p1-center)) / np.linalg.norm(p2 - p1))
                if d < closestDist:
                    print(i, d)
                    closestLine = i
                    closestDist = d
            self.detectedNotes[closestLine].append(rect)



    def _nonMaxSupp(self, pts, img):
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
        for i in range(len(self.lines)):
            line = self.lines[i]
            if self.detectedHolds[i]:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            cv.line(img, (line[0], line[1]), (line[2], line[3]), color, 3, cv.LINE_AA)

        for targetStat in self.targetStats:
            [x, y, w, h, a] = targetStat
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        for i in range(len(self.detectedNotes)):
            for rect in self.detectedNotes[i]:
                x, y, w, h = rect
                cv.rectangle(img, (x, y), (x+w, y+h), note_colors[i], 3)
