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

def drawPoints(img, ptArr, rvecs, tvecs, mtx, dist):
  ptArr_C = cv.projectPoints(ptArr, rvecs, tvecs, mtx, dist)[0]
  # print(ptArr_C)
  color = tuple([255, 255, 0])
  for pt_C in ptArr_C:
    img = cv.circle(img, tuple(pt_C.ravel()), 3, color, -1)
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


def calcLineByS(s, o, a):
  p = (1 - s) * o + s * a
  return np.array(p)


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

coordTestArr_C = [[1100, 800, 1]]

cap = []
for i in range(0, len(TARGET)):
  cap.append(cv.VideoCapture(CAMPORT[i]))

imgs = [[], []]

# while loop (for debug)
while True:
  for i in range(0, len(TARGET)):
    imgs[i] = cv.undistort(cap[i].read()[1], mtx[i], dist[i])
    imgs[i] = drawVector(imgs[i], origin, axis, rvecs[i], tvecs[i], mtx[i], dist[i])
  a = calcWorldCoordinate(coordTestArr_C[0], rvecs[1], tvecs[1], mtx[1])
  o = calcWorldCoordinate([0, 0, 0], rvecs[1], tvecs[1], mtx[1])
  # print('a below')
  # print(a)
  # print('o below')
  # print(o)
  ptArr = []
  # for tmp in range(18, 19, 0.1):
  for tmp in np.arange(17, 18, 0.1):
    ptArr.append(calcLineByS(tmp, o, a))
  ptArr = np.float32(np.array(ptArr)).reshape(-1, 3)
  print(ptArr)
  imgs[1] = drawPoints_C(imgs[1], coordTestArr_C)
  imgs[0] = drawPoints(imgs[0], ptArr, rvecs[0], tvecs[0], mtx[0], dist[0])
  cv.imshow('img_' + TARGET[0], imgs[0])
  cv.imshow('img_' + TARGET[1], imgs[1])
  if cv.waitKey(1) & 0xFF == ord('q'):
      break



for i in range(0, len(TARGET)):
  cap[i].release()
cv.destroyAllWindows()

