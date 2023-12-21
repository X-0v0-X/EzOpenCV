import ezcv
import cv2

try:
    ezcv.useYuNet()  # 顔認識の方法を変える
    while True:
        frame = ezcv.getFrame()
        frame = ezcv.drawRectToFace(frame)
        frame = ezcv.drawRectToEyes(frame)
        frame = ezcv.drawRibbonToFace(frame)
        cv2.imshow("", frame)

        # ESCキーを押すと終了
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    ezcv.release()
    cv2.destroyAllWindows()
    exit()
except KeyboardInterrupt:
    ezcv.release()
    cv2.destroyAllWindows()
    exit()

# ezcv.useYuNet()
# while True:
#     frame = ezcv.getFrame()
#     frame = ezcv.drawRectToFace(frame)
#     frame = ezcv.drawRectToEyes(frame)
#     frame = ezcv.drawRibbonToFace(frame)
#     cv2.imshow("", frame)
#     cv2.waitKey(1)
