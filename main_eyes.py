import cv2
import numpy as np

LOCAL_IP = "localhost"
LOCAL_PORT = "11111"
ADDR = "udp://" + LOCAL_IP + ":" + LOCAL_PORT 

def toGray(frame):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grayFrame

def loadCascade():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    left_cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
    right_cacade = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
    return face_cascade , left_cascade , right_cacade

def getFaceList(frame, cascade):
    faceList = cascade.detectMultiScale(frame, minSize=(200, 200))
    return faceList

def getEyesList(frame,faces,left_cascade,right_cascade):
    left_eyes = []
    right_eyes = []
    for x1,y1,w1,h1 in faces:
        left_eyes_frame = frame[y1:int(y1+h1/2),int(x1+w1/2):x1+w1]
        right_eyes_frame = frame[y1:int(y1+h1/2),x1:int(x1+w1/2)]
        left_eyes = left_cascade.detectMultiScale(left_eyes_frame)
        right_eyes = right_cascade.detectMultiScale(right_eyes_frame)
    return left_eyes , right_eyes
    
def drawRectToEyes(frame, faces , left_eyes , right_eyes): 
    for x1,y1,w1,h1 in faces:   
        if len(left_eyes) > 0 :
            for x2,y2,w2,h2 in left_eyes:
                cv2.rectangle(frame,(int(x1+w1/2+x2),y1+y2),(int(x1+w1/2+x2+w2),y1+y2+h2),(255,0,0),1)
        if len(right_eyes) > 0:
            for x3,y3,w3,h3 in right_eyes:
                cv2.rectangle(frame,(x1+x3,y1+y3),(x1+x3+w3,y1+y3+h3),(0,255,0),1)
    return frame

def drawRectToFace(frame, faces):
    if len(faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
    return frame



def loadRibbon():
    ribbon = cv2.imread("ribbon-sm.png", -1)
    return ribbon

def drawImgToFrame(frame: np.array, img: np.array, x, y):
    imgH, imgW = img.shape[:2]
    frmH, frmW = frame.shape[:2]
    x-=int(imgW/2)
    y-=int(imgH/2)
    ax1, ax2 = 0, imgW
    ay1, ay2 = 0, imgH
    x1, x2 = x, x+imgW
    y1, y2 = y, y+imgH
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
        for (x, y, w, h) in faces:
            frame = drawImgToFrame(frame, ribbon, x+w, y)
    return frame

try:
    face_cascade , left_cascade , right_cascade = loadCascade()
    ribbon = loadRibbon()

    while(True):
        cap = cv2.VideoCapture(ADDR)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0.1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        if cap.isOpened():
            break

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            faces = getFaceList(toGray(frame), face_cascade)
            left_eyes , right_eyes = getEyesList(toGray(frame), faces,left_cascade,right_cascade)
            frame = drawRectToFace(frame, faces)
            frame = drawRectToEyes(frame,faces,left_eyes, right_eyes)
            frame = drawRibbonToFace(frame, faces, ribbon)
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    exit()