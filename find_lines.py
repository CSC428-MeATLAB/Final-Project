"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np

# color ranges of targets
color_ranges = [
    [np.array([54,127,63]), np.array([64,255,198])],    # green
    [np.array([-4,127,77]), np.array([4,255,190])],     # red
    [np.array([26,127,77]), np.array([36,255,190])],    # yellow
    [np.array([102,127,77]), np.array([109,255,190])],  # blue
    [np.array([14,127,68]), np.array([20,255,198])]     # orange
]

def main(argv):
    
    default_file = 'sudoku.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    im_bgr = cv.imread(cv.samples.findFile(filename))
    dimensions = im_bgr.shape

    # Check if image is loaded fine
    if im_bgr is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    im_gray = cv.cvtColor(im_bgr, cv.COLOR_BGR2GRAY)
    im_hsv = cv.cvtColor(im_bgr, cv.COLOR_BGR2HSV)
    
    dst = cv.Canny(im_gray, 125, 150, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 60, None, 100, 10)
    # throw out horizontal lines
    isVertical = np.abs(linesP[:,0:,0] - linesP[:,0:,2]) < np.abs(linesP[:,0:,1] - linesP[:,0:,3])
    linesP = linesP[isVertical]

    lines = []
    line_start_y = 0
    line_end_y = dimensions[0]
    for color_range in color_ranges:
        target, stats = find_target(im_hsv, color_range[0], color_range[1])
        [x, y, w, h, a] = stats
        cv.rectangle(cdstP, (x, y), (x+w, y+h), (0, 255, 0), 3)
        target[1] -= h / 2

        line = find_closest_line(target, linesP)
        # make sure lower point comes first
        if line[1] < line[3]:
            line = np.roll(line, 2)
        
        line_start_y += y - h / 4 * 3
        if line[3] < line_end_y:
            line_end_y = line[3]

        lines.append(line)

    line_start_y = line_start_y / len(lines)

    # adjust lines to only cover string
    for line in lines:
        ys = line[1] - line[3]
        xs = line[0] - line[2]
        t = (line_start_y - line[1]) / ys
        line[0] = round(line[0] + xs * t)
        line[1] = round(line_start_y)

        t = (line_end_y - line[1]) / ys
        line[2] = round(line[0] + xs * t)
        line[3] = round(line_end_y)
    
    for line in lines:
        cv.line(cdstP, (line[0], line[1]), (line[2], line[3]), (0,0,255), 3, cv.LINE_AA)

    holding = cv.cvtColor(cv.imread("holding2.png"), cv.COLOR_BGR2HSV)

    for i in range(len(lines)):
        line = lines[i]
        samples = []
        for j in range(10):
            t = j / 100
            x = int(round(line[0] + t * (line[2] - line[0])))
            y = int(round(line[1] + t * (line[3] - line[1])))
            samples.append(holding[y,x])
        samples = np.array(samples)
        std = np.average(np.std(samples, axis=0))
        avg = np.average(samples, axis=0)
        if std < 10 and avg[1] > 100:
            print("Held note detected: " + str(i))

    for line in lines:
        cv.line(holding, (line[0], line[1]), (line[2], line[3]), (0,0,255), 3, cv.LINE_AA)

    cv.imwrite("test.png", holding)
    cv.imwrite("canny.png", dst)
    cv.imwrite("houghp.png", cdstP)

    return 0

def find_closest_line(pt, lines):
    dist2 = np.minimum(
        (lines[:,0] - pt[0])**2 + (lines[:,1] - pt[1])**2,
        (lines[:,2] - pt[0])**2 + (lines[:,3] - pt[1])**2
    )
    order = np.argsort(dist2)

    return lines[order[0]]

def find_target(img, hsv_low, hsv_high):
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
    
if __name__ == "__main__":
    main(sys.argv[1:])