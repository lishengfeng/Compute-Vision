import cv2

start_x, start_y = -1, -1
end_x, end_y = -1, -1
paste_x, paste_y = -1, -1

drawing = False  # True if mouse is pressed
rows = None
columns = None


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

    elif event == cv2.EVENT_RBUTTONDOWN:
        paste_x, paste_y = x, y


if __name__ == '__main__':
    img_name = 'Eagle.jpg'
    source_img = cv2.imread(img_name)
    clone_image = source_img.copy()
    rows, columns = source_img.shape[:2]
    # print(rows)
    # print(columns)
    cv2.namedWindow(img_name)
    cv2.setMouseCallback(img_name, draw_circle)
    roi = None
    while 1:
        cv2.imshow(img_name, clone_image)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # 27 -> ESC
            break
        elif k == ord('c'):
            # Do copy
            if start_x < 0 or start_y < 0 or end_x < 0 or end_y < 0:
                continue
            else:
                if start_y > end_y:
                    start_y, end_y = end_y, start_y
                if start_x > end_x:
                    start_x, end_x = end_x, start_x
                print(start_x, start_y, end_x, end_y)
                roi = source_img[start_y:end_y, start_x:end_x, :]
                print(roi.shape)
        elif k == ord('p'):
            # Do paste
            if roi is not None and paste_x >= 0 and paste_y >= 0:
                roi_rows = roi.shape[0]
                roi_columns = roi.shape[1]
                y_offset = roi_rows
                y_offset = y_offset if y_offset + paste_y < rows else rows \
                                                                       - \
                                                                       paste_y
                x_offset = roi_columns
                x_offset = x_offset if x_offset + paste_x < columns else \
                    columns - paste_x
                clone_image[paste_y: paste_y + y_offset,
                paste_x: paste_x + x_offset] = roi[0:y_offset, 0:x_offset]

    cv2.destroyAllWindows()
