#
# カメラの歪みを補正する
#
import numpy as np
import cv2
import glob
from time import sleep
from datetime import datetime

TMP_FOLDER_PATH = "./res_labs_camera/"
MTX_PATH = TMP_FOLDER_PATH + "mtx.csv"
DIST_PATH = TMP_FOLDER_PATH + "dist.csv"
SAVE_FOLDER_PATH = "./aftercalib_labs_camera/"

# メイン関数
def main():
    calibrateImage() # 画像の歪みを補正

# カメラの歪みをCSVファイルを元に補正する関数
def calibrateImage():
    mtx, dist = loadCalibrationFile(MTX_PATH, DIST_PATH)

    for fn in glob.glob("./img_labs_camera/*"):
        img = cv2.imread(fn)
        resultImg = cv2.undistort(img, mtx, dist, None) # 内部パラメータを元に画像補正
        saveImgByTime(SAVE_FOLDER_PATH, resultImg)
        sleep(1)

# キャリブレーションCSVファイルを読み込む関数
def loadCalibrationFile(mtx_path, dist_path):
    try:
        mtx = np.loadtxt(mtx_path, delimiter=',')
        dist = np.loadtxt(dist_path, delimiter=',')
    except Exception as e:
        raise e
    return mtx, dist

# 画像を時刻で保存する関数
def saveImgByTime(dirPath, img):
    # 時刻を取得
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = dirPath + date + ".png"
    cv2.imwrite(path, img) # ファイル保存
    print("saved: ", path)

if __name__ == '__main__':
    main()