import cv2
import numpy as np


def mySobelEdgeDetector(img, gHalfWinSize=7, gSigma=1, thresh=0.5):
    # Perform a Gaussian smoothing on the given image
    if (gHalfWinSize != 0 and gHalfWinSize % 2 == 0) or gHalfWinSize < 0:
        raise ValueError
    ksize = 0 if gHalfWinSize == 0 else (gHalfWinSize - 1) // 2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_s = cv2.GaussianBlur(gray, (ksize, ksize), gSigma)
    # Create two Sobel filters in both x and y directions
    sobelx64f = cv2.Sobel(img_s, cv2.CV_64F, 1, 0, ksize=ksize)
    abs_sobelx64f = np.absolute(sobelx64f)

    sobely64f = cv2.Sobel(img_s, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobely64f = np.absolute(sobely64f)
    M, ang = cv2.cartToPolar(abs_sobelx64f, abs_sobely64f)
    # M, ang = cv2.cartToPolar(sobelx64f, sobely64f)
    norm_M = cv2.normalize(M, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    ret, thresh_img = cv2.threshold(norm_M, thresh, 255, cv2.THRESH_BINARY)
    cv2.imwrite('MySovel.png', thresh_img)
    return thresh_img


if __name__ == '__main__':
    # source_img = cv2.imread('TestEdge1.jpg')
    # clone_image = source_img.copy()
    # res = mySobelEdgeDetector(clone_image)
    source_img = cv2.imread('TestEdge2.jpg')
    clone_image_1 = source_img.copy()
    res_1 = mySobelEdgeDetector(clone_image_1)

    while 1:
        # cv2.imshow('Edge Detection', res)
        cv2.imshow('Edge Detection 2', res_1)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # 27 -> ESC
            break
    cv2.destroyAllWindows()
