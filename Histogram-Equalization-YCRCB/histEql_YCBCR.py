import cv2
import numpy as np


def myHEQ_YCRCB(img):
    Y, Cr, Cb = cv2.split(img)
    equ_Y = cv2.equalizeHist(Y)
    equ = cv2.merge((equ_Y, Cr, Cb))
    res = np.hstack((img, equ))
    cv2.imwrite('HEQ_YCRCB.png', equ)
    return res


if __name__ == '__main__':
    source_img = cv2.imread('Castle_badexposure.jpg')
    clone_image = source_img.copy()
    # source_img = cv2.imread('UnderExposedScene.jpg')
    # clone_image_1 = source_img.copy()
    imgYCC = cv2.cvtColor(clone_image, cv2.COLOR_BGR2YCR_CB)

    while 1:
        cv2.imshow('HEQ_YCRCB', myHEQ_YCRCB(imgYCC))
        # cv2.imshow('HEQ_RGB_1', myHEQ_RGB(clone_image_1))
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # 27 -> ESC
            break
    cv2.destroyAllWindows()
