import cv2
import numpy as np
import queue
import threading


class EzCV:
    __instance = None

    def _toGray(self, frame):
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return grayFrame

    def loadCascade(self):
        cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        lEyeCascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
        rEyeCascade = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
        return cascade, lEyeCascade, rEyeCascade

    def loadModel(self):
        # model = cv2.FaceDetectorYN.create("yunet_n_640_640.onnx", "", (0, 0))
        model = cv2.FaceDetectorYN.create(
            "face_detection_yunet_2023mar.onnx", "", (0, 0)
        )
        return model

    def _getFaceList(self, frame):
        faceList = self.cascade.detectMultiScale(frame, minSize=(100, 100))
        return faceList

    def _getFaceListYN(self, frame):
        resized = cv2.resize(frame, None, fx=0.5, fy=0.5)
        _, faceList = self.model.detect(resized)
        faceList = (
            [
                [
                    int(element[0] * 2),
                    int(element[1] * 2),
                    int(element[2] * 2),
                    int(element[3] * 2),
                ]
                for element in faceList
            ]
            if faceList is not None
            else []
        )
        return faceList

    def _drawRectToFace(self, frame, faces):
        if len(faces):
            for x, y, w, h in faces:
                # print(x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
        return frame

    def _getEyesList(self, frame, faces):
        left_eyes = []
        right_eyes = []
        for x1, y1, w1, h1 in faces:
            left_eyes_frame = frame[y1 : int(y1 + h1 / 2), int(x1 + w1 / 2) : x1 + w1]
            right_eyes_frame = frame[y1 : int(y1 + h1 / 2), x1 : int(x1 + w1 / 2)]
            left_eyes = self.left_cascade.detectMultiScale(left_eyes_frame)
            right_eyes = self.right_cascade.detectMultiScale(right_eyes_frame)
        return left_eyes, right_eyes

    def _drawRectToEyes(self, frame, faces, left_eyes, right_eyes):
        for x1, y1, w1, h1 in faces:
            if len(left_eyes) > 0:
                for x2, y2, w2, h2 in left_eyes:
                    cv2.rectangle(
                        frame,
                        (int(x1 + w1 / 2 + x2), y1 + y2),
                        (int(x1 + w1 / 2 + x2 + w2), y1 + y2 + h2),
                        (255, 0, 0),
                        1,
                    )
            if len(right_eyes) > 0:
                for x3, y3, w3, h3 in right_eyes:
                    cv2.rectangle(
                        frame,
                        (x1 + x3, y1 + y3),
                        (x1 + x3 + w3, y1 + y3 + h3),
                        (0, 255, 0),
                        1,
                    )
        return frame

    def loadRibbon(self):
        ribbon = cv2.imread("ribbon-sm.png", -1)
        return ribbon

    def _drawImgToFrame(self, frame: np.array, img: np.array, x, y):
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

    def _drawRibbonToFace(self, frame, faces):
        if len(faces):
            for x, y, w, h in faces:
                frame = self._drawImgToFrame(frame, self.ribbon, x + w, y)
        return frame

    def _capture(self):
        print("waiting")
        # while True:
        #     if self.cap.isOpened():
        #         ret, _ = self.cap.read()
        #         if ret:
        #             print("received")
        #             break
        #     cv2.waitKey(1)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.videoQueue.empty():
                try:
                    self.videoQueue.get_nowait()
                except queue.Empty:
                    pass
            if not self.tempQueue.empty():
                try:
                    self.tempQueue.get_nowait()
                except queue.Empty:
                    pass
            self.videoQueue.put(frame)
            self.tempQueue.put(frame)
            cv2.waitKey(1)

    def _read(self):
        return self.videoQueue.get()

    def _readTemp(self):
        return self.tempQueue.get()

    def release(self):
        self.cap.release()

    def _faceDetect(self):
        while True:
            frame = self._readTemp()
            if self.frameCount > 5:
                self.frameCount = 0
                if self.useYuNet:
                    self.faceList = self._getFaceListYN(frame)
                    self.leftEyes, self.rightEyes = self._getEyesList(
                        frame, self.faceList
                    )
                else:
                    self.faceList = self._getFaceList(frame)
                    self.leftEyes, self.rightEyes = self._getEyesList(
                        frame, self.faceList
                    )
            self.frameCount += 1
            cv2.waitKey(1)

    def __init__(
        self, ip="192.168.10.1", port="11111", width=1280, height=720, useYuNet=False
    ):
        print("init")
        EzCV.__instance = self
        address = "udp://" + ip + ":" + port + "?overrun_nonfatal=1&fifo_size=50000000"
        self.cap = cv2.VideoCapture(address)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("H", "2", "6", "4"))
        self.videoQueue = queue.Queue()
        self.tempQueue = queue.Queue()
        self.cascade, self.left_cascade, self.right_cascade = self.loadCascade()
        self.model = self.loadModel()
        self.ribbon = self.loadRibbon()
        self.useYuNet = useYuNet
        self.model.setInputSize((int(width / 2), int(height / 2)))

        self.faceList = []
        self.leftEyes = []
        self.rightEyes = []
        self.frameCount = 0

        capThread = threading.Thread(target=self._capture)
        capThread.daemon = True
        while not capThread.is_alive():
            capThread.start()

        detectThread = threading.Thread(target=self._faceDetect)
        detectThread.daemon = True
        while not detectThread.is_alive():
            detectThread.start()

    @staticmethod
    def getInstance():
        if EzCV.__instance is None:
            EzCV()
        return EzCV.__instance


def useYuNet():
    ezcv = EzCV.getInstance()
    ezcv.useYuNet = True


def drawRectToFace(frame):
    ezcv = EzCV.getInstance()
    faces = ezcv.faceList
    return ezcv._drawRectToFace(frame, faces)


def drawRibbonToFace(frame):
    ezcv = EzCV.getInstance()
    faces = ezcv.faceList
    return ezcv._drawRibbonToFace(frame, faces)


def drawRectToEyes(frame):
    ezcv = EzCV.getInstance()
    faces = ezcv.faceList
    leftEyes = ezcv.leftEyes
    rightEyes = ezcv.rightEyes
    return ezcv._drawRectToEyes(frame, faces, leftEyes, rightEyes)


def faceDetect():
    ezcv = EzCV.getInstance()
    ezcv._faceDetect()


def getFrame():
    ezcv = EzCV.getInstance()
    return ezcv._read()


def release():
    ezcv = EzCV.getInstance()
    ezcv.release()
