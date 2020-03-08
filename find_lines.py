"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np

def main(argv):
    
    default_file = 'sudoku.png'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    im_bgr = cv.imread(cv.samples.findFile(filename))

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

    color_ranges = [
        [np.array([-4,127,77]), np.array([4,255,190])],
        [np.array([26,127,77]), np.array([36,255,190])],
        [np.array([14,127,68]), np.array([20,255,198])],
        [np.array([54,127,63]), np.array([64,255,198])],
        [np.array([102,127,77]), np.array([109,255,190])]
    ]
    for color_range in color_ranges:
        blob, stats = find_blob(im_hsv, color_range[0], color_range[1])
        [x, y, w, h, a] = stats
        cv.rectangle(cdstP, (x, y), (x+w, y+h), (0, 255, 0), 3)
        blob[1] -= h / 2

        l = linesP[0][0]
        closestD = min(
            (l[0]-blob[0])**2 + (l[1]-blob[1])**2,
            (l[2]-blob[0])**2 + (l[3]-blob[1])**2,
        )
        closestLine = l
        for i in range(1, len(linesP)):
            l = linesP[i][0]
            d = min(
                (l[0]-blob[0])**2 + (l[1]-blob[1])**2,
                (l[2]-blob[0])**2 + (l[3]-blob[1])**2,
            )
            if d < closestD:
                closestD = d
                closestLine = l

        l = closestLine
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    if linesP is not None and False:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            d = math.sqrt((l[0]-l[2])**2 + (l[1] - l[3])**2)
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    # cv.imshow("Source", src)
    cv.imwrite("canny.png", dst)
    cv.imwrite("houghp.png", cdstP)
    
    cv.waitKey()
    return 0

def find_blob(img, hsv_low, hsv_high):
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