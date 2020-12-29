#
# カメラ画像をキャプチャする
#
import cv2
import time
from datetime import datetime

TARGET = "left_cam"
cap = cv2.VideoCapture(2) # 任意のカメラ番号に変更する
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
start_time = 0

while True:
    start_time = time.time()
    ret, frame = cap.read()
    cv2.imshow("camera", frame)

    k = cv2.waitKey(1)&0xff # キー入力を待つ
    if k == ord('p'):
        # 「p」キーで画像を保存
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = "./img_" + TARGET + "/" + date + ".png"
        cv2.imwrite(path, frame) # ファイル保存

        # cv2.imshow(path, frame) # キャプチャした画像を表示
    elif k == ord('q'):
        # 「q」キーが押されたら終了する
        break

    elapsed_time = time.time() - start_time
    # print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()