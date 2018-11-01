# import the necessary packages
import numpy as np
import cv2
import math
import sys



def detection_of_lensflare(file):

    # load the image
    image = cv2.imread(str(file))

    # blur the image to make it easier to detect objects
    blur_image = cv2.medianBlur(image, 3)

    # convert the image to grayscale
    gray = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)

    # set the lightness threshold to filter the lenflare (small blob)
    ret, thresh2 = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    img = thresh2

    # use canny to detect the lens flare edge
    imgray = cv2.Canny(img, 600, 100, 3)

    # set the threshold
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

    # find the lensflare contour
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#contours为轮廓集，可以计算轮廓的长度、面积等
    for cnt in contours:
        # set the fit number
        if len(cnt) > 150:
            S1 = cv2.contourArea(cnt)
            ell = cv2.fitEllipse(cnt)
            S2 = math.pi*ell[1][0]*ell[1][1]
            if (S1/S2) > 0.2 and ell[1][0] / ell[1][1] < 2 and ell[1][1] / ell[1][0] < 2:
                # lensflare
                return 1
    # no lensflare
    return 0

def main():
    # receive the image name
    file = sys.argv[1]

    # read image
    img1 = cv2.imread(file)

    # convert the image to hue saturation lightness saturation model
    hls = cv2.cvtColor(img1, cv2.COLOR_BGR2HLS)

    # split the h l s value
    h, l, s = cv2.split(hls)

    # if the mean lightness is bigger than 150
    if np.mean(l) > 150:

        # lensflare
        return 1

    # use detection_of_lensflare model to detect
    else:
        detection_of_lensflare(file)

if __name__ == '__main__':
    main()