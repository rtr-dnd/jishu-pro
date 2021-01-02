import numpy as np
import math
import serial
import time

ser = serial.Serial("/dev/tty.usbserial-14130", 9600)
time.sleep(10)

count = 0
step = np.linspace(0, 2 * math.pi, 10)
# ser.write(bytes('F00v', 'utf-8'))
# print('sent plus')
# time.sleep(5)
# ser.write(bytes('mF00v', 'utf-8'))
# print('sent minus')
# time.sleep(5)
# ser.write(bytes('s', 'utf-8'))
while True:
  count += 1
  if count >= len(step):
    count = 0
  
  # print('---')
  # print(step[count])
  # print(math.sin(step[count]))
  speed = 0xA000 * (math.sin(step[count]))
  # print(hex(int(speed))[2:].upper() + 'v')
  print('a' * count)
  if speed < 0:
    ser.write(bytes('m' + hex(int(-speed))[2:].upper() + 'v', 'utf-8'))
  else:
    ser.write(bytes(hex(int(speed))[2:].upper() + 'v', 'utf-8'))

  time.sleep(0.05) # 20fps confirmed

ser.close()