import csv
import numpy as np
import cv2 as cv

TARGET = ["left_cam", "right_cam"]
CAMPORT = [0, 2]

mtx = np.zeros((len(TARGET), 3, 3))
dist = np.zeros((len(TARGET), 5))
rvecs = np.zeros((len(TARGET), 3))
tvecs = np.zeros((len(TARGET), 3))

# 記載なきときはワールド座標、Cがついていたらカメラ座標

def findMarkers(img):
  hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
  mask1 = cv.inRange(hsv_img, (0, 120, 70), (5, 255, 255))
  mask2 = cv.inRange(hsv_img, (175, 120, 70), (180, 255, 255))
  mask = cv.bitwise_or(mask1, mask2)
  masked_img = cv.bitwise_and(img, img, mask=mask)
  contours, hierarchy = cv.findContours(mask, 1, 2)
  contours.sort(key=lambda s: len(s))
  cnt1 = contours[-1] # largest contour
  cnt2 = contours[-2] #2nd largest contour

  # simplified contour
  # epsilon = 0.05 * cv.arcLength(cnt, True)
  epsilon = 1

  approx1 = cv.approxPolyDP(cnt1, epsilon, True)
  M1 = cv.moments(approx1)
  approx_center1 = [int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]), 1] # gravity center of the largest contour

  approx2 = cv.approxPolyDP(cnt2, epsilon, True)
  M2 = cv.moments(approx2)
  approx_center2 = [int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]), 1] # gravity center of the 2nd largest contour

  # masked_img = cv.drawContours(masked_img, [approx], 0, (0, 255, 0), 5)
  # masked_img = cv.circle(masked_img, tuple(approx_center), 3, (0, 255, 0), -1)
  # print(approx)
  if (approx_center1[1] < approx_center2[1]): # スクリーン座標で上にあるやつが一番大きいとき
    return [approx_center1, approx_center2] # 0番目: ペンの上側マーカー 1番目: 下側マーカー
  else:
    return [approx_center2, approx_center1]

def drawVector(img, originArr, goalArr, rvecs, tvecs, mtx, dist):
  # corner = tuple(corners[0].ravel()) ## ワールド座標原点（二次元）
  origin_C = cv.projectPoints(originArr, rvecs, tvecs, mtx, dist)[0][0]
  goalArr_C = cv.projectPoints(goalArr, rvecs, tvecs, mtx, dist)[0]

  # print('goalArr below')
  # print(goalArr)
  '''
  [[ 3.  0.  0.]
  [ 0.  3.  0.]
  [ 0.  0. -3.]]
  '''

  for goalIdx in range(0, 3):
    if goalIdx == 0:
      img = cv.line(img, tuple(origin_C.ravel()), tuple(goalArr_C[goalIdx].ravel()), (255,0,0), 5)
    elif goalIdx == 1:
      img = cv.line(img, tuple(origin_C.ravel()), tuple(goalArr_C[goalIdx].ravel()), (0,255,0), 5)
    else:
      img = cv.line(img, tuple(origin_C.ravel()), tuple(goalArr_C[goalIdx].ravel()), (0,0,255), 5)
    # 0: 長辺、1: 短辺、2: 垂直
    # x: long, y: short, z: vertical

  return img

def drawPoints(img, ptArr, rvecs, tvecs, mtx, dist, color):
  ptArr_C = cv.projectPoints(ptArr, rvecs, tvecs, mtx, dist)[0]
  # print(ptArr_C)
  for pt_C in ptArr_C:
    img = cv.circle(img, tuple(pt_C.ravel()), 10, color, -1)
  return img

def drawPoints_C(img, ptArr_C):
  color = tuple([0, 255, 255])
  for pt_C in ptArr_C:
    img = cv.circle(img, tuple([pt_C[0], pt_C[1]]), 5, color, -1)
  return img

def drawPointsRaw(img, x, y, z, rvecs, tvecs, mtx):
  vec = np.float32(np.array([x, y, z, 1]))
  tr_mat = np.column_stack([cv.Rodrigues(rvecs)[0], tvecs])
  pt_C = np.dot(np.matmul(mtx, tr_mat), vec)
  pt_C = np.float32(pt_C / pt_C[2])
  # print(pt_C)
  img = cv.circle(img, (pt_C[0], pt_C[1]), 5, (0, 0, 255), -1)
  return img

def calcWorldCoordinate(pt_C, rvecs, tvecs, mtx):
  # pt_C: 正規化済みのスクリーン座標点 u, v, 1
  pt_CC = np.dot(np.linalg.inv(mtx), pt_C) # x', y', 1 のカメラ座標点
  res = np.dot(np.linalg.inv(cv.Rodrigues(rvecs)[0]), pt_CC - tvecs)
  return res

def calcLine(t, o, n):
  return o + t * n


# read parameters
for i in range(0, len(TARGET)):
  with open('calibration/res_' + TARGET[i] + '/mtx_' + TARGET[i] + '.csv') as f:
    reader = csv.reader(f)
    mtx[i] = [row for row in reader]
  with open('calibration/res_' + TARGET[i] + '/dist_' + TARGET[i] + '.csv') as f:
    reader = csv.reader(f)
    for row in reader:
      dist[i] = row
  rvecs[i] = np.loadtxt('calibration/pose_' + TARGET[i] + '/rvecs_' + TARGET[i] + '.csv')
  tvecs[i] = np.loadtxt('calibration/pose_' + TARGET[i] + '/tvecs_' + TARGET[i] + '.csv')

mtx = np.float32(np.array(mtx))
dist = np.float32(np.array(dist))
dist = dist[..., np.newaxis]

origin = np.float32([[0, 0, 0]]).reshape(-1,3)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
testPoints = np.float32([[5, 0, 0], [0, 0, -3]]).reshape(-1, 3)

coordTestArr_C = [[650, 290, 1], [850, 490, 1]] # (u, v, 1)

cap = []
for i in range(0, len(TARGET)):
  cap.append(cv.VideoCapture(CAMPORT[i]))

imgs = [[], []]
red_centers = [[], []] # i番目: i番目のカメラに映るマーカーの中心座標×2（スクリーン座標u, v, 1）
o = [[], []] # i番目: i番目のカメラ座標で(0, 0, 0)の世界座標

a1 = [[], []] # i番目: i番目のカメラ座標で(x', y', 1)となる点の世界座標（上マーカー）
a2 = [[], []] # i番目: i番目のカメラ座標で(x', y', 1)となる点の世界座標（下マーカー）
n1 = [[], []] # i番目: i番目のカメラから目標点までの正規化済み方向ベクトル（世界座標）（上マーカー）
n2 = [[], []] # i番目: i番目のカメラから目標点までの正規化済み方向ベクトル（世界座標）（下マーカー）
d1 = [[], []] # i番目: i番目のカメラからの最近点までの距離（世界座標）（上マーカー）
d2 = [[], []] # i番目: i番目のカメラからの最近点までの距離（世界座標）（下マーカー）
ptArr1 = [[], []] # i番目: i番目のカメラから見たときある点に見えるような線の集合（世界座標）（上マーカー）
ptArr2 = [[], []] # i番目: i番目のカメラから見たときある点に見えるような線の集合（世界座標）（下マーカー）

while True:
  # 直線を出す用
  for i in range(0, len(TARGET)):
    imgs[i] = cv.undistort(cap[i].read()[1], mtx[i], dist[i])
    red_centers[i] = findMarkers(imgs[i])

    # 直線を計算するために必要な情報
    o[i] = calcWorldCoordinate([0, 0, 0], rvecs[i], tvecs[i], mtx[i]) # カメラ座標で(0, 0, 0)の世界座標
    a1[i] = calcWorldCoordinate(red_centers[i][0], rvecs[i], tvecs[i], mtx[i]) #  カメラ座標で(x', y', 1)の世界座標
    a2[i] = calcWorldCoordinate(red_centers[i][1], rvecs[i], tvecs[i], mtx[i]) #  カメラ座標で(x', y', 1)の世界座標
    n1[i] = (a1[i] - o[i]) / np.linalg.norm(a1[i] - o[i]) # 方向の単位ベクトル
    n2[i] = (a2[i] - o[i]) / np.linalg.norm(a2[i] - o[i]) # 方向の単位ベクトル
 
  # 直線から空間上の点を出す用
  for i in range(0, len(TARGET)):
    num1 = np.dot(n1[i], o[not(i)] - o[i]) # 分子
    num2 = np.dot(n1[i], n1[not(i)]) * np.dot(n1[not(i)], o[not(i)] - o[i]) # 分子
    den = 1 - np.dot(n1[i], n1[not(i)]) ** 2 # 分母
    d1[i] = (num1 - num2) / den
    ptArr1[i] = np.float32(np.array(o[i] + d1[i] * n1[i])).reshape(-1, 3)

    num1 = np.dot(n2[i], o[not(i)] - o[i]) # 分子
    num2 = np.dot(n2[i], n2[not(i)]) * np.dot(n2[not(i)], o[not(i)] - o[i]) # 分子
    den = 1 - np.dot(n2[i], n2[not(i)]) ** 2 # 分母
    d2[i] = (num1 - num2) / den
    ptArr2[i] = np.float32(np.array(o[i] + d2[i] * n2[i])).reshape(-1, 3)
  
  # print(ptArr1)
  avgpt1 = np.average(ptArr1, axis=0)
  avgpt2 = np.average(ptArr2, axis=0)
  print(avgpt1)
  print(avgpt2)

  # 描画用
  for i in range(0, len(TARGET)):
    # imgs[i] = drawPoints_C(imgs[i], [coordTestArr_C[i]])
    imgs[i] = drawPoints(imgs[i], avgpt1, rvecs[i], tvecs[i], mtx[i], dist[i], (255, 255, 0))
    imgs[i] = drawPoints(imgs[i], avgpt2, rvecs[i], tvecs[i], mtx[i], dist[i], (0, 255, 255))
    # imgs[i] = cv.circle(imgs[i], tuple(red_centers[0]), 10, (100, 255, 0), -1)
    # imgs[i] = cv.circle(imgs[i], tuple(red_centers[1]), 10, (100, 0, 255), -1)
    imgs[i] = drawVector(imgs[i], origin, axis, rvecs[i], tvecs[i], mtx[i], dist[i])
    # cv.imshow('img_' + TARGET[i], imgs[i])

  if cv.waitKey(1) & 0xFF == ord('q'):
      break

for i in range(0, len(TARGET)):
  cap[i].release()
cv.destroyAllWindows()

