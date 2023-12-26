import cv2
import numpy as np
import queue
import threading
import time

LOCAL_IP = "localhost"
LOCAL_PORT = "11111"
ADDR = "udp://" + LOCAL_IP + ":" + LOCAL_PORT

videoQueue = queue.Queue()


def toGray(frame):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grayFrame


def loadCascade():
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return cascade


def loadModel():
    # model = cv2.FaceDetectorYN.create("yunet_n_640_640.onnx", "", (0, 0))
    model = cv2.FaceDetectorYN.create("yunet_n_640_640.onnx", "", (0, 0))
    return model


def setModelInputSize(frame, model):
    h, w = frame.shape[:2]
    model.setInputSize((w, h))
    return model


def getFaceList(frame, cascade):
    faceList = cascade.detectMultiScale(frame, minSize=(100, 100))
    return faceList


def getFaceListYN(frame, model):
    _, faceList = model.detect(frame)
    faceList = (
        [
            [int(element[0]), int(element[1]), int(element[2]), int(element[3])]
            for element in faceList
        ]
        if faceList is not None
        else []
    )
    return faceList


def drawRectToFace(frame, faces):
    if len(faces):
        for x, y, w, h in faces:
            # print(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
    return frame


def loadRibbon():
    ribbon = cv2.imread("ribbon-sm.png", -1)
    return ribbon


def drawImgToFrame(frame: np.array, img: np.array, x, y):
    imgH, imgW = img.shape[:2]
    frmH, frmW = frame.shape[:2]
    x -= int(imgW / 2)
    y -= int(imgH / 2)
    ax1, ax2 = 0, imgW
    ay1, ay2 = 0, imgH
    x1, x2 = x, x + imgW
    y1, y2 = y, y + imgH
    if x1 < 0:
        ax1 = ax1 - x1
        x1 = 0
    if x2 > frmW:
        ax2 = ax2 - (x2 - frmW)
        x2 = frmW
    if y1 < 0:
        ay1 = ay1 - y1
        y1 = 0
    if y2 > frmH:
        ay2 = ay2 - (y2 - frmH)
        y2 = frmH

    mask = img[ay1:ay2, ax1:ax2, 3]
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask / 255
    img = img[ay1:ay2, ax1:ax2, :3]
    frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - mask)
    frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] + img * mask
    frame[y1:y2, x1:x2]
    return frame


def drawRibbonToFace(frame, faces, ribbon):
    if len(faces):
        for x, y, w, h in faces:
            frame = drawImgToFrame(frame, ribbon, x + w, y)
    return frame


def capture():
    while True:
        if cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if not videoQueue.empty():
                try:
                    videoQueue.get_nowait()
                except queue.Empty:
                    pass
            videoQueue.put(frame)


try:
    cascade = loadCascade()
    model = loadModel()
    ribbon = loadRibbon()

    print("wait")
    while True:
        cap = cv2.VideoCapture(ADDR)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("H", "2", "6", "4"))
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                model = setModelInputSize(frame, model)
                break

    t = threading.Thread(target=capture)
    t.start()

    while cap.isOpened():
        frame = videoQueue.get()
        # faces = getFaceList(toGray(frame), cascade)
        faces = getFaceListYN(frame, model)
        frame = drawRectToFace(frame, faces)
        frame = drawRibbonToFace(frame, faces, ribbon)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    exit()
