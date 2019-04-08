import cv2
import numpy as np


def init_images(img_filename1, img_filename2):
    img1 = cv2.imread(img_filename1)
    img2 = cv2.imread(img_filename2)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1_gray, img2_gray


def padding(img, pad_size):
    old_shape = img.shape[:2]
    new_shape = [old_shape[0] + pad_size * 2, old_shape[1] + pad_size * 2]
    nImg = np.zeros(new_shape, dtype=np.float32)
    nImg[pad_size:old_shape[0] + pad_size, pad_size:old_shape[1] + pad_size] = img
    return nImg


def NMS(img, thresh, half_win_size):
    n_img = padding(img, half_win_size)
    h = n_img.shape[0]
    w = n_img.shape[1]
    corners = []
    for i in range(half_win_size, h - half_win_size):
        for j in range(half_win_size, w - half_win_size):
            if n_img[i, j] > thresh:
                done = False
                for a in range(i - half_win_size, i + half_win_size + 1):
                    for b in range(j - half_win_size, j + half_win_size + 1):
                        if n_img[i, j] < n_img[a, b]:
                            done = True
                            break
                    if done:
                        break
                if not done:
                    corners.append((i - half_win_size, j - half_win_size))
    return corners


def construct_openCVKeyPtList(corners, keyptsize=1):
    keyPtsList = []
    for i in range(len(corners)):
        # corners[i][1] and corners[i][0] store the column and row index of the corner
        keyPtsList.append(cv2.KeyPoint(x=corners[i][1], y=corners[i][0], _size=keyptsize))
    return keyPtsList


def construct_openCVDMatch(corners1, corners2, cornerMatches):
    dmatch = list()
    for i in range(len(cornerMatches)):
        # cornerMatches[i][0] and cornerMatches[i][1] store the indices of corresponded corners, respectively
        # cornerMatches[i][2] stores the corresponding ZNCC matching score
        c_match = cv2.DMatch(cornerMatches[i][0], cornerMatches[i][1], cornerMatches[i][2])
        dmatch.append(c_match)
    return dmatch


def draw_matches(img1, img2, corners1, corners2, matches):
    keyPts1 = construct_openCVKeyPtList(corners1)
    keyPts2 = construct_openCVKeyPtList(corners2)
    dmatch = construct_openCVDMatch(corners1, corners2, matches)
    matchingImg = cv2.drawMatches(img1, keyPts1, img2, keyPts2, dmatch, None)
    cv2.imwrite('cornerMatching.png', matchingImg)


def save_corners(img, corners, filename):
    cloneImg = img.copy()
    for (y, x) in corners:
        cv2.circle(cloneImg, (x, y), 3, (0, 0, 128), 2)
    cv2.imwrite(filename, cloneImg)


def show_corners(img, corners):
    cloneImg = img.copy()
    for (y, x) in corners:
        cv2.circle(cloneImg, (x, y), 3, (0, 0, 128), 2)
    cv2.imshow("corners", cloneImg)
    # k = cv2.waitKey(k0)


def extract_keypts_Harris(img, thresh_Harris=0.05, nms_size=15):
    rImg = cv2.cornerHarris(img, blockSize=5, ksize=5, k=0.04)
    corners = NMS(rImg, thresh_Harris, nms_size)
    return corners


def score_ZNCC(patch1, patch2):
    v1 = patch1.flatten()
    v2 = patch2.flatten()
    if v1.size != v2.size:
        return -1
    nv1 = v1 - np.mean(v1)
    nv2 = v2 - np.mean(v2)
    norm1 = np.linalg.norm(nv1)
    norm2 = np.linalg.norm(nv2)
    if norm1 == 0 or norm2 == 0:
        return -1
    nv1 = nv1 / norm1
    nv2 = nv2 / norm2
    score = np.dot(nv1, nv2)
    return score


def matchKeyPts(img1, img2, corners1, corners2, patchSize=15, max_score_thresh=0.98):
    match_list = []
    nImg1 = padding(img1, patchSize)
    nImg2 = padding(img2, patchSize)
    for idx1, c1 in enumerate(corners1):
        max_match_score = 0
        best_match = -1
        for idx2, c2 in enumerate(corners2):
            y1 = c1[0]
            x1 = c1[1]
            y2 = c2[0]
            x2 = c2[1]
            patch1 = nImg1[y1:y1 + 2 * patchSize + 1, x1:x1 + 2 * patchSize + 1]
            patch2 = nImg2[y2:y2 + 2 * patchSize + 1, x2:x2 + 2 * patchSize + 1]
            match_score = score_ZNCC(patch1, patch2)
            if match_score > max_match_score:
                max_match_score = match_score
                best_match = idx2
        if max_match_score > max_score_thresh:
            match_list.append((idx1, best_match, max_match_score))
    return match_list


def main():
    img1, img2 = init_images("goldengate-02.png", "goldengate-03.png")
    corners1 = extract_keypts_Harris(img1)
    save_corners(img1, corners1, "corner1.png")
    corners2 = extract_keypts_Harris(img2)
    save_corners(img2, corners2, "corners2.png")
    match_list = matchKeyPts(img1, img2, corners1, corners2)
    draw_matches(img1, img2, corners1, corners2, match_list)


if __name__ == "__main__":
    main()
