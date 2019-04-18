import cv2
import numpy as np
import math


def load_correspondingKeypts(inFileName):
    with open(inFileName, 'r') as infile:
        line1 = [int(x) for x in next(infile).split()]
        listSize = line1[0]
        corners1 = []
        corners2 = []
        for line in infile:
            line = [int(x) for x in line.split()]
            corners1.append((line[0], line[1]))
            corners2.append((line[2], line[3]))
    # print(listSize)
    # print(len(corners1))    
    # print(len(corners2))    
    return corners1, corners2


def solveHomography(corners1, corners2):
    # Follow the slides matrix
    n = len(corners1)
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    for i in range(n):
        xp = corners1[i][1]
        yp = corners1[i][0]
        x = corners2[i][1]
        y = corners2[i][0]
        A[i * 2, 0] = x
        A[i * 2, 1] = y
        A[i * 2, 2] = 1
        A[i * 2, 6] = -x * xp
        A[i * 2, 7] = -y * xp
        A[i * 2 + 1, 3] = x
        A[i * 2 + 1, 4] = y
        A[i * 2 + 1, 5] = 1
        A[i * 2 + 1, 6] = -x * yp
        A[i * 2 + 1, 7] = -y * yp
        b[i * 2] = xp
        b[i * 2 + 1] = yp
    h = np.linalg.lstsq(A, b, rcond=None)[0]

    h_mat = np.ones((3, 3))
    h_mat[0, 0] = h[0]
    h_mat[0, 1] = h[1]
    h_mat[0, 2] = h[2]
    h_mat[1, 0] = h[3]
    h_mat[1, 1] = h[4]
    h_mat[1, 2] = h[5]
    h_mat[2, 0] = h[6]
    h_mat[2, 1] = h[7]

    return h_mat


def saveHomography(outFileName, h):
    with open(outFileName, 'w') as ofile:
        ofile.write('{0:.2f} {1:.2f} {2:.2f}\n'.format(h[0, 0], h[0, 1], h[0, 2]))
        ofile.write('{0:.2f} {1:.2f} {2:.2f}\n'.format(h[1, 0], h[1, 1], h[1, 2]))
        ofile.write('{0:.2f} {1:.2f} {2:.2f}\n'.format(h[2, 0], h[2, 1], h[2, 2]))
    ofile.close()


def computeShift(stitchRange):
    t_x = math.ceil(-min(stitchRange[0, 0], 0))
    t_y = math.ceil(-min(stitchRange[1, 0], 0))
    stitchRange[0, 1] += t_x
    stitchRange[1, 1] += t_y
    stitchRange[0, 0] = 0
    stitchRange[1, 0] = 0
    return t_x, t_y


def blendImages(img1, img2, h, t_x, t_y, stitchRange):
    range_x = int(math.ceil(stitchRange[0, 1]))
    range_y = int(math.ceil(stitchRange[1, 1]))
    nImg = np.zeros((range_y, range_x, 3), dtype=np.uint8)

    img1_RSize, img1_CSize = img1.shape[:2]
    img2_RSize, img2_CSize = img2.shape[:2]
    inv_H = np.linalg.inv(h)

    for x in range(range_x):
        for y in range(range_y):
            x1 = x - t_x
            y1 = y - t_y
            q = np.array([x1, y1, 1])
            p = transformByH(inv_H, q)
            inImg1 = True
            inImg2 = True
            x2 = int(round(p[0]))
            y2 = int(round(p[1]))
            if x1 < 0 or y1 < 0 or x1 >= img1_CSize or y1 >= img1_RSize:
                inImg1 = False
            if x2 < 0 or y2 < 0 or x2 >= img2_CSize or y2 >= img2_RSize:
                inImg2 = False
            if inImg1 and inImg2:
                for i in range(3):
                    color1 = int(round(img1.item(y1, x1, i) / 2))
                    color2 = int(round(img2.item(y2, x2, i) / 2))
                    nImg.itemset((y, x, i), color1 + color2)
            elif inImg1 and not inImg2:
                for i in range(3):
                    nImg.itemset((y, x, i), img1.item(y1, x1, i))
            elif not inImg1 and inImg2:
                for i in range(3):
                    nImg.itemset((y, x, i), img2.item(y2, x2, i))
            else:
                pass

    return nImg


def transformByH(h, p):
    q = h.dot(p)
    if -1e-5 < q[2] < 1e-5:
        raise ValueError("Devided by zero.")
    q[0] = q[0] / q[2]
    q[1] = q[1] / q[2]
    return q


def computeRangeOfStitchedImage(h, r1Size, c1Size, r2Size, c2Size):
    i2Range = np.zeros((2, 2))
    p = np.array([0, 0, 1])
    for y in range(r2Size):
        p[0] = 0
        p[1] = y
        q = transformByH(h, p)
        if q[0] < i2Range[0, 0]:
            i2Range[0, 0] = q[0]
        if q[0] > i2Range[0, 1]:
            i2Range[0, 1] = q[0]
        if q[1] < i2Range[1, 0]:
            i2Range[1, 0] = q[1]
        if q[1] > i2Range[1, 1]:
            i2Range[1, 1] = q[1]
        p[0] = c2Size - 1
        p[1] = y
        q = transformByH(h, p)
        if q[0] < i2Range[0, 0]:
            i2Range[0, 0] = q[0]
        if q[0] > i2Range[0, 1]:
            i2Range[0, 1] = q[0]
        if q[1] < i2Range[1, 0]:
            i2Range[1, 0] = q[1]
        if q[1] > i2Range[1, 1]:
            i2Range[1, 1] = q[1]
    stitchRange = np.zeros((2, 2))
    stitchRange[0, 1] = c1Size - 1
    stitchRange[1, 1] = r1Size - 1
    for i in range(2):
        stitchRange[i, 0] = min(stitchRange[i, 0], i2Range[i, 0])
        stitchRange[i, 1] = max(stitchRange[i, 1], i2Range[i, 1])
    return stitchRange


def main():
    corners1, corners2 = load_correspondingKeypts("matchedCorners.txt")
    h = solveHomography(corners1, corners2)
    # saveHomography("myhomography.txt", h)
    img1 = cv2.imread("goldengate-02.png")
    img2 = cv2.imread("goldengate-03.png")
    r1Size, c1Size = img1.shape[0:2]
    r2Size, c2Size = img2.shape[0:2]
    stitchRange = computeRangeOfStitchedImage(h, r1Size, c1Size, r2Size, c2Size)
    # print ("Range of stitched image" + str(stitchRange))
    t_x, t_y = computeShift(stitchRange)
    nImg = blendImages(img1, img2, h, t_x, t_y, stitchRange)
    cv2.imwrite("stitchedImg.png", nImg)


if __name__ == "__main__":
    main()
