import numpy as np
import cv2
import time

if __name__ == '__main__':
    video = cv2.VideoCapture(0)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print(
            "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        print(
            "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    num_frames = 120
    print("Capturing {0} frames".format(num_frames))
    start = time.time()

    for i in range(0, num_frames):
        ret, frame = video.read()

    end = time.time()

    seconds = end - start
    print("Time taken: {0} seconds".format(seconds))
    print("fps: {0}".format(num_frames/seconds))

    video.release()

    # while(True):
    #     ret, frame = cap.read()
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     cv2.imshow('frame', gray)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
