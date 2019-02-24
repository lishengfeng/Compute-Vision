import cv2
import numpy as np


def myHEQ_RGB(img):
    b, g, r = cv2.split(img)
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ = cv2.merge((equ_b, equ_g, equ_r))
    res = np.hstack((img, equ))
    cv2.imwrite('HEQ_RGB.png', equ)
    return res


if __name__ == '__main__':
    source_img = cv2.imread('Castle_badexposure.jpg')
    clone_image = source_img.copy()
    # source_img = cv2.imread('UnderExposedScene.jpg')
    # clone_image_1 = source_img.copy()
    res = myHEQ_RGB(clone_image)
    while 1:
        cv2.imshow('HEQ_RGB', res)
        # cv2.imshow('HEQ_RGB_1', myHEQ_RGB(clone_image_1))
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # 27 -> ESC
            break
    cv2.destroyAllWindows()
