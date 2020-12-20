import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
print('onlydraw')
print('import finished')

img1 = cv.imread('test_img/pen_1.jpg', 1)  # queryimage # left image
img2 = cv.imread('test_img/pen_2.jpg', 1)  # trainimage # right image
print('images read')

F = np.array([[ 8.92265017e-06, -2.31697122e-05, 2.11373885e-03],
     [-1.60881962e-06,  3.44321463e-05, -1.24695378e-02],
    [-5.43492914e-03, 8.38793471e-03, 1.00000000e+00]])

def findRedAreaCenter(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv_img, (0, 120, 70), (5, 255, 255))
    mask2 = cv.inRange(hsv_img, (175, 120, 70), (180, 255, 255))
    mask = cv.bitwise_or(mask1, mask2)
    masked_img = cv.bitwise_and(img, img, mask=mask)
    contours, hierarchy = cv.findContours(mask, 1, 2)
    contours.sort(key=lambda s: len(s))
    cnt = contours[-1] # largest contour

    # minimum rectangle contour
    # rect = cv.minAreaRect(cnt)
    # box = cv.boxPoints(rect)
    # box = np.int0(box)
    # print(box)
    # box_center = np.round(np.mean(box, axis=0)).astype(int)
    # print(box_center)
    # masked_img = cv.drawContours(masked_img, [box], 0, (0, 255, 0), 5)
    # masked_img = cv.circle(masked_img, tuple(box_center), 3, (0, 255, 0), -1)

    # simplified contour
    epsilon = 0.1 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    M = cv.moments(approx)
    approx_center = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])] # gravity center of the contour
    # masked_img = cv.drawContours(masked_img, [approx], 0, (0, 255, 0), 5)
    # masked_img = cv.circle(masked_img, tuple(approx_center), 3, (0, 255, 0), -1)
    # print(approx)
    return approx_center

center1 = findRedAreaCenter(img1)
center2 = findRedAreaCenter(img2)
center_img2 = cv.circle(img2, tuple(center2), 3, (0, 255, 0), -1)

# pts2 = np.array([[330, 280], [430, 450]])
pts2 = np.array([center2])

# def drawlines(img1, img2, lines, pts1, pts2):
def drawlines(color_img1, color_img2, lines, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    img1 = cv.cvtColor(color_img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(color_img2, cv.COLOR_BGR2GRAY)
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    # for r, pt1, pt2 in zip(lines, pts1, pts2): # 線の下図はポイントの数と一致
    for r, pt2 in zip(lines, pts2): # 線の下図はポイントの数と一致
        # color = tuple(np.random.randint(0, 255, 3).tolist())
        color = (0, 0, 255)
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 3)
        # img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 10, color, -1)
    return img1, img2

lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
# lines: ax + by + c = 0 の(a, b, c)
lines1 = lines1.reshape(-1, 3)

img5, img6 = drawlines(img1, img2, lines1, pts2)

plt.subplot(221), plt.imshow(img5)
plt.subplot(222), plt.imshow(img6)
plt.subplot(223), plt.imshow(cv.cvtColor(center_img2, cv.COLOR_BGR2RGB))
# plt.subplot(223), plt.imshow(mask)
plt.subplot(224), plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
plt.show()
print('plt done')
