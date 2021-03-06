import csv
import numpy as np
import cv2 as cv
import copy
import serial
import threading
import time
# from pynput import keyboard

TARGET = ["right_cam", "left_cam"]
CAMPORT = [0, 2]
Z_OFFSET = 0.1 # 天板との接地点のz座標（世界座標）
MOTOR_UNIT = 9650 # 1マス分のモーターステップ数
MOTOR_MARGIN = 2.0 # モーターの可動域（何マスか）

temp_for_measure = 0
start_time = 0

cur_pos = 0 # 今のモータ座標位置

ser = serial.Serial("/dev/tty.usbserial-14130", 9600)

# is_shiftkey_pressed = False

# def on_press(key):
#   try:
#     print('alphanumeric key {0} pressed'.format(
#       key.char))
#   except AttributeError:
#     print('special key {0} pressed'.format(
#       key))
#   if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
#     is_shiftkey_pressed = True

# def on_release(key):
#   if key == keyboard.Key.shift or key == keyboard.Key.shift_r:
#     is_shiftkey_pressed = False
#   if key == keyboard.Key.esc:
#       # Stop listener
#       return False

# listener = keyboard.Listener(
#     on_press=on_press,
#     on_release=on_release)
# listener.start()

mtx = np.zeros((len(TARGET), 3, 3))
dist = np.zeros((len(TARGET), 5))
rvecs = np.zeros((len(TARGET), 3))
tvecs = np.zeros((len(TARGET), 3))
# 記載なきときはワールド座標、Cがついていたらカメラ座標

def findMarkers(img):
  hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
  # mask1 = cv.inRange(hsv_img, (0, 160, 70), (5, 255, 255))
  # mask2 = cv.inRange(hsv_img, (175, 160, 70), (180, 255, 255))
  # mask = cv.bitwise_or(mask1, mask2)
  mask = cv.inRange(hsv_img, (50, 100, 50), (86, 255, 255))
  masked_img = cv.bitwise_and(img, img, mask=mask)
  contours, hierarchy = cv.findContours(mask, 1, 2)
  contours.sort(key=lambda s: len(s))
  if (len(contours) < 2):
    return [False]
  cnt1 = contours[-1] # largest contour
  cnt2 = contours[-2] #2nd largest contour

  # simplified contour
  # epsilon = 0.05 * cv.arcLength(cnt, True)
  epsilon = 1

  approx1 = cv.approxPolyDP(cnt1, epsilon, True)
  M1 = cv.moments(approx1)
  if (M1["m00"] == 0.0):
    return [False]
  approx_center1 = [int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]), 1] # gravity center of the largest contour

  approx2 = cv.approxPolyDP(cnt2, epsilon, True)
  M2 = cv.moments(approx2)
  if (M2["m00"] == 0.0):
    return [False]
  approx_center2 = [int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]), 1] # gravity center of the 2nd largest contour

  # masked_img = cv.drawContours(masked_img, [approx], 0, (0, 255, 0), 5)
  # masked_img = cv.circle(masked_img, tuple(approx_center), 3, (0, 255, 0), -1)
  # print(approx)
  if (approx_center1[1] < approx_center2[1]): # スクリーン座標で上にあるやつが一番大きいとき
    return [True, [approx_center1, approx_center2]] # 0番目: ペンの上側マーカー 1番目: 下側マーカー
  else:
    return [True, [approx_center2, approx_center1]]

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
    img = cv.circle(img, tuple(pt_C.astype(int).ravel()), 10, color, -1)
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

def calcContactPoint(upper, lower):
  x0 = ((upper[2] - Z_OFFSET) * lower[0] - (lower[2] - Z_OFFSET) * upper[0]) / (upper[2] - lower[2])
  y0 = ((upper[2] - Z_OFFSET) * lower[1] - (lower[2] - Z_OFFSET) * upper[1]) / (upper[2] - lower[2])
  return np.float32(np.array([x0, y0, Z_OFFSET]))

def sendPos(pos):
  hex_val = ''
  if (pos >= 0):
    hex_val = hex(pos)
  else:
    hex_val = hex((1<<22) + pos)
  ser.write(bytes(hex_val[2:].upper() + 'z', 'utf-8'))

def sendVel(vel):
  hex_val = ''
  if (vel >= 0):
    hex_val = hex(vel)[2:].upper()
  else:
    # hex_val = 'm' + hex((1<<20) + vel)[2:].upper()
    hex_val = 'm' + hex(vel)[3:].upper()
  print('sendvel: ' + str(hex_val + 'v'))
  ser.write(bytes(hex_val + 'v', 'utf-8'))

# Ku = MOTOR_UNIT * 70
# Pu = 0.5
# Kp = MOTOR_UNIT * 70 * 0.6 * 0.5
K = 3.0
Kp = MOTOR_UNIT * 5 * K
# Ti = 0.5 * Pu
Ki = MOTOR_UNIT * 0.1 * 0.6 / (0.5 * 0.3) * K
# Ki = 0
# Td = 0.125 * Pu
Kd = MOTOR_UNIT * 0.1 * 0.6 / (0.5 * 0.3) / 4 * K
# Kd = 0
diff = [0.0, 0.0]
integral = 0.0
prev_time_pid = time.time()

def pid(sensor_cp, target_cp):
  global Kp, Ki, Kd, diff, integral
  global prev_time_pid
  
  delta_t_pid = time.time() - prev_time_pid
  prev_time_pid = time.time()

  diff[0] = diff[1]
  diff[1] = - (sensor_cp[1] - target_cp[1])
  # cpは下がプラス→s時点より下の方にあるとsensor - targetがプラス→下に動かしたい
  # dを押すとマイナスが送られるのでsendvelではマイナス
  integral += (diff[0] + diff[1]) / 2 * delta_t_pid

  p = Kp * diff[1]
  i = Ki * integral
  d = Kd * (diff[1] - diff[0]) / delta_t_pid

  print('diff: ' + str(diff[1]))
  print('p: ' + str(p))
  print('i: ' + str(i))
  print('d: ' + str(d))
  res = max(min(p + i + d, 180000), -180000)
  if abs(res) < 1000:
    res = 0
  print('res: ' + str(res))
  return res


def fasterForLoop(ser):
  global cur_pos
  global cur_destination
  while(True):
    tmp = repr(ser.readline().decode())
    try:
      cur_pos = int(tmp[1:-5])
      # print(cur_pos)
      if (destination != []):
        cur_destination[1] = destination[1] - (cur_pos - dest_pos) / MOTOR_UNIT # todo: 何かがおかしい
    except:
      pass
      # print(tmp)

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

cap = []
for i in range(0, len(TARGET)):
  cap.append(cv.VideoCapture(CAMPORT[i]))
  # 高速化
  cap[i].set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('H', '2', '6', '4'))
  cap[i].set(cv.CAP_PROP_FRAME_WIDTH, 1280)
  cap[i].set(cv.CAP_PROP_FRAME_HEIGHT, 720)
  cap[i].set(cv.CAP_PROP_BUFFERSIZE, 1)

imgs = [[], []]
red_centers = [[], []] # i番目: i番目のカメラに映るマーカーの中心座標×2（スクリーン座標u, v, 1）
markersDetected = False
o = [[], []] # i番目: i番目のカメラ座標で(0, 0, 0)の世界座標

a1 = [[], []] # i番目: i番目のカメラ座標で(x', y', 1)となる点の世界座標（上マーカー）
a2 = [[], []] # i番目: i番目のカメラ座標で(x', y', 1)となる点の世界座標（下マーカー）
n1 = [[], []] # i番目: i番目のカメラから目標点までの正規化済み方向ベクトル（世界座標）（上マーカー）
n2 = [[], []] # i番目: i番目のカメラから目標点までの正規化済み方向ベクトル（世界座標）（下マーカー）
d1 = [[], []] # i番目: i番目のカメラからの最近点までの距離（世界座標）（上マーカー）
d2 = [[], []] # i番目: i番目のカメラからの最近点までの距離（世界座標）（下マーカー）
ptArr1 = [[], []] # i番目: i番目のカメラから見たときある点に見えるような線の集合（世界座標）（上マーカー）
ptArr2 = [[], []] # i番目: i番目のカメラから見たときある点に見えるような線の集合（世界座標）（下マーカー）
target_pos = 0 # 起動したときを0とした絶対位置（ステップ数）
destination = [] # shiftを押し始めたときのcontact point
cur_destination = np.float32(np.array([0, 0, 0])) # destinationに相当する紙上の点が今世界座標でどこにあるか
prev_cp = []
dest_pos = 0 # shiftを押し始めたときのモータ座標
motor_loop_interval = 0
prev_time = time.time() # 前回ループ時の時刻
smooth_cp = [] # ローパスフィルタ済みの値
param_a = 0.6 # ローパスフィルタ用係数

blank_image = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
cv.imshow('blank', blank_image)

thread = threading.Thread(target=fasterForLoop, args=(ser,))
thread.setDaemon(True)
thread.start()


while True:
  start_time = time.time()
  # 直線を出す用
  for i in range(0, len(TARGET)):
    imgs[i] = cv.undistort(cap[i].read()[1], mtx[i], dist[i])
    
    temp = findMarkers(imgs[i])
    if (temp[0]):
      markersDetected = True
    else:
      markersDetected = False
      break
    red_centers[i] = temp[1]

    # 直線を計算するために必要な情報
    o[i] = calcWorldCoordinate([0, 0, 0], rvecs[i], tvecs[i], mtx[i]) # カメラ座標で(0, 0, 0)の世界座標
    a1[i] = calcWorldCoordinate(red_centers[i][0], rvecs[i], tvecs[i], mtx[i]) #  カメラ座標で(x', y', 1)の世界座標
    a2[i] = calcWorldCoordinate(red_centers[i][1], rvecs[i], tvecs[i], mtx[i]) #  カメラ座標で(x', y', 1)の世界座標
    n1[i] = (a1[i] - o[i]) / np.linalg.norm(a1[i] - o[i]) # 方向の単位ベクトル
    n2[i] = (a2[i] - o[i]) / np.linalg.norm(a2[i] - o[i]) # 方向の単位ベクトル
  
  if (not(markersDetected)):
    print('markers not detected')
    continue

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
  cp = calcContactPoint(avgpt1.ravel(), avgpt2.ravel())
  if (smooth_cp == []):
    smooth_cp = copy.deepcopy(cp)
  else:
    smooth_cp = (1 - param_a) * smooth_cp + param_a * cp
  # print(cp)
  # motor_loop_interval -= 1
  # if (motor_loop_interval < 0):
  # print(cp)
  # motor_loop_interval = 3
  # if (destination == []):
  #   print('destination not set')
    # continue
  # else:
  velocity = 0 # 毎回0にする
  if (destination != []):
    # dif = cp[1] - cur_destination[1]
    # target_pos = int(cur_pos - dif * MOTOR_UNIT)
    # sendPos(target_pos)
    tick_length = time.time() - prev_time
    if (tick_length < 0.5): # fail safe
      # velocity = int(-(cp[1] - prev_cp[1]) * MOTOR_UNIT / tick_length)
      velocity = int(pid(smooth_cp, cur_destination))
      # print(tick_length)
      # print(velocity)
      # print(hex(velocity))
    prev_time = time.time()
    prev_cp = cp
  sendVel(velocity)
  
  # 描画用
  # for i in range(0, len(TARGET)):
    # imgs[i] = drawPoints(imgs[i], avgpt1, rvecs[i], tvecs[i], mtx[i], dist[i], (255, 255, 0))
    # imgs[i] = drawPoints(imgs[i], avgpt2, rvecs[i], tvecs[i], mtx[i], dist[i], (0, 255, 255))
    # imgs[i] = cv.circle(imgs[i], tuple(red_centers[i][0][0:2]), 10, (100, 255, 0), -1)
    # imgs[i] = cv.circle(imgs[i], tuple(red_centers[i][1][0:2]), 10, (100, 0, 255), -1)
    # imgs[i] = drawPoints(imgs[i], np.array([smooth_cp]), rvecs[i], tvecs[i], mtx[i], dist[i], (255, 0, 255))
    # imgs[i] = drawPoints(imgs[i], np.array([cur_destination]), rvecs[i], tvecs[i], mtx[i], dist[i], (255, 255, 0))
    # imgs[i] = drawVector(imgs[i], origin, axis, rvecs[i], tvecs[i], mtx[i], dist[i])
    # cv.imshow('img_' + TARGET[i], imgs[i])
  cv.imshow('blank', blank_image)


  k = cv.waitKey(1)
  # print(k)
  if k != -1:
    if k == ord('s'): # set destination
      prev_time = time.time()
      destination = copy.deepcopy(cp)
      cur_destination = copy.deepcopy(cp)
      dest_pos = cur_pos
      target_pos = cur_pos
      prev_cp = cp
      print('set')
      print(destination)
    elif k == ord('f'): # follow destination
      if (destination == []):
        print('destination not set')
        continue
      print('destination below')
      print(destination)
      print('cur_destination below')
      print(cur_destination)
      dif = cp[1] - cur_destination[1]
      target_pos = int(cur_pos - dif * MOTOR_UNIT)
      print('target_pos below')
      print(target_pos)
      sendPos(target_pos)
    elif k == ord('x'):
      print('sending val x')
      sendPos(9600)
    elif k == ord('y'):
      print('sending val y')
      sendPos(-9600)
    elif k == ord('r'): # reset
      print('reset')
      destination = []
      ser.write(bytes('r', 'utf-8'))
    elif k == ord('u'): # up、奥
      print('up')
      ser.write(bytes('u', 'utf-8'))
    elif k == ord('v'):
      sendVel(0x5000)
    elif k == ord('n'):
      sendVel(20400)
    elif k == ord('m'):
      sendVel(-20400)
    elif k == ord('b'):
      destination = []
      sendVel(0)
    elif k == ord('c'): # up、奥
      destination = []
      print('clear destination')
    elif k == ord('d'): # up、奥
      print('up')
      ser.write(bytes('d', 'utf-8'))
    elif k == ord('h'): # up、奥
      print('home')
      ser.write(bytes('h', 'utf-8'))
    elif k == ord('q'):
      sendVel(0)
      break
  
  # temp_for_measure += 1
  # if (temp_for_measure >= 5000):
  #   break

  elapsed_time = time.time() - start_time
  print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
  print('---------')

sendVel(0)
for i in range(0, len(TARGET)):
  cap[i].release()
cv.destroyAllWindows()
ser.close()