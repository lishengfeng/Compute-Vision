import cv2
import numpy as np

start_x, start_y = -1, -1
end_x, end_y = -1, -1
paste_x, paste_y = -1, -1

drawing = False  # True if mouse is pressed
rows = None
columns = None
roi = None


def myHEQ_YCRCB():
    global clone_image, start_x, start_y, end_x, end_y, roi
    img_ori = clone_image.copy()
    imgYCC = cv2.cvtColor(img_ori, cv2.COLOR_BGR2YCR_CB)
    Y, Cr, Cb = cv2.split(imgYCC)
    equ_Y = cv2.equalizeHist(Y)
    equ = cv2.merge((equ_Y, Cr, Cb))

    if start_y > end_y:
        start_y, end_y = end_y, start_y
    if start_x > end_x:
        start_x, end_x = end_x, start_x
    roi = equ[start_y:end_y, start_x:end_x, :]
    clone_image[start_y:end_y, start_x:end_x] = roi[:, :]
    res = np.hstack((img_ori, clone_image))
    cv2.imwrite('HEQ_ROI.png', clone_image)
    return res


def draw_rectangle(x1, y1, x2, y2):
    global clone_image, end_x, end_y
    clone_image = source_img.copy()
    x2 = x2 if x2 <= columns - 1 else columns - 1
    y2 = y2 if y2 <= rows - 1 else rows - 1
    cv2.line(clone_image, (x1, y1), (x1, y2), (0, 255, 0))
    cv2.line(clone_image, (x1, y1), (x2, y1), (0, 255, 0))
    cv2.line(clone_image, (x1, y2), (x2, y2), (0, 255, 0))
    cv2.line(clone_image, (x2, y1), (x2, y2), (0, 255, 0))
    end_x = x2
    end_y = y2


def draw_circle(event, x, y, flags, param):
    global start_x, start_y, paste_x, paste_y, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            draw_rectangle(start_x, start_y, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        # try to drag the mouse before you release the left button
        drawing = False
        myHEQ_YCRCB()

    elif event == cv2.EVENT_RBUTTONDOWN:
        paste_x, paste_y = x, y


if __name__ == '__main__':
    source_img = cv2.imread('NikonContest2016Winner.png')
    clone_image = source_img.copy()
    rows, columns = source_img.shape[:2]
    # source_img = cv2.imread('UnderExposedScene.jpg')
    # clone_image_1 = source_img.copy()
    window_name = 'HEQ_ROI'
    cv2.namedWindow(window_name)
    imgYCC = cv2.cvtColor(clone_image, cv2.COLOR_BGR2YCR_CB)
    cv2.setMouseCallback(window_name, draw_circle)
    while 1:
        cv2.imshow(window_name, clone_image)
        # cv2.imshow('HEQ_RGB_1', myHEQ_RGB(clone_image_1))
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # 27 -> ESC
            break
    cv2.destroyAllWindows()
