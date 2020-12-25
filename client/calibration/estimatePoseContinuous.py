import csv
import numpy as np
import cv2 as cv
import glob

TARGET = "right_cam"
# Load previously saved data
mtx = []
dist = []
with open('res_' + TARGET + '/mtx_' + TARGET + '.csv') as f:
  reader = csv.reader(f)
  mtx = [row for row in reader]
with open('res_' + TARGET + '/dist_' + TARGET + '.csv') as f:
  reader = csv.reader(f)
  for row in reader:
    dist = row
mtx = np.float32(np.array(mtx))
dist = np.float32(np.array(dist))
dist = dist[..., np.newaxis]
print(mtx)
print(dist)
print(np.shape(dist))

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel()) ## ワールド座標原点（二次元）
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def drawPoints(img, imgpts):
  color = tuple([255, 0, 0])
  for imgpt in imgpts:
    img = cv.circle(img, tuple(imgpt[0]), 5, color, -1)
  return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)
objp = objp[..., np.newaxis]
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
testPoint = np.float32([[5, 0, 0], [0, 0, -3]]).reshape(-1, 3)

cap1 = cv.VideoCapture(2)

while True:
  ret, img = cap1.read()
  img = cv.undistort(img, mtx, dist)
  # img = cv.imread('./img_labs_camera/repr.png')
  gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
  ret, corners = cv.findChessboardCorners(gray, (10, 7),None)
  if ret == True:
      corners2 = np.squeeze(cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria))
      corners2 = corners2[..., np.newaxis]
      # Find the rotation and translation vectors.
      ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
      # project 3D points to image plane
      # imgpts: 単位ベクトルのx,y×3
      imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
      img = draw(img,corners2,imgpts)
      imgpts, jac = cv.projectPoints(testPoint, rvecs, tvecs, mtx, dist)
      img = drawPoints(img, imgpts)
      cv.imshow('img_' + TARGET, img)
      if cv.waitKey(1) & 0xFF == ord('q'):
        print(ret)
        print(rvecs)
        print(tvecs)
        np.savetxt('./pose_' + TARGET + '/rvecs_' + TARGET + '.csv', rvecs, delimiter=', ', fmt="%0.14f")
        np.savetxt('./pose_' + TARGET + '/tvecs_' + TARGET + '.csv', tvecs, delimiter=', ', fmt="%0.14f")
        break
  else:
    print('not found')

cap1.release()
cv.destroyAllWindows()
