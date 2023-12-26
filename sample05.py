import tello
import cv2
import threading
import os
import keyboard
import numpy as np
import ezcv

lr = 0  # 左右 左がマイナス、右がプラス
fb = 0  # 前後 後ろがマイナス、前がプラス
ud = 0  # 上下　下がマイナス

takePhotoFlag = False  # 写真撮影のボタンが押されているか判断する変数

flyingFlag = True  # 映像表示を終了するための変数
videoMode = 0


# ▼関数（一連の命令をまとめて実行してくれるもの）。def 関数名():という風に書く
def cameraThread():
    global takePhotoFlag
    try:
        while True:
            frame = ezcv.getFrame()
            frame = ezcv.drawRectToFace(frame)
            frame = ezcv.drawRectToEyes(frame)
            frame = ezcv.drawRibbonToFace(frame)

            if videoMode == 1:  # 虹色モード
                frame = ezcv.niji(frame)

            cv2.imshow("ドローンカメラ(ESCキーを押すと終了)", frame)  # ドローンから送られてきた映像をPCに映す
            # ESCキーを押すと終了
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            if flyingFlag == False:  # ドローンの飛行が終了した場合
                break

            if takePhotoFlag == True:  # 写真撮影のボタンが押されていた場合
                ezcv.takePicture(frame)
                takePhotoFlag = False  # 写真撮影のボタンをリセット

        ezcv.release()  # 映像を停止する
        cv2.destroyAllWindows()  # 映像を映すためのウィンドウを閉じる
        exit()  # プログラムを終了する

    except KeyboardInterrupt:
        ezcv.release()  # 映像を停止する
        cv2.destroyAllWindows()  # 映像を映すためのウィンドウを閉じる
        exit()  # プログラムを終了する


try:
    print("program start")
    tello.Send("command")
    print("battery", tello.get_battery())
    tello.Send("streamon")  # telloでカメラを使うためのコマンド

    _cameraThread = threading.Thread(target=cameraThread)  # キーボード操作と、映像表示を同時に行うための命令
    while not _cameraThread.is_alive():
        _cameraThread.start()  # キーボード操作と、映像表示を同時に行うための命令

    # メインループ
    print("メインループ開始")
    while True:
        if keyboard.is_pressed("space"):
            tello.Send("takeoff")

        if keyboard.is_pressed("shift"):
            tello.Send("land")

        if keyboard.is_pressed("up"):  # ↑キーが押されているとき
            if fb <= 30:
                fb += 1
        elif keyboard.is_pressed("down"):  # ↓キーが押されているとき
            if fb >= -30:
                fb -= 1
        else:  # ↑キーと↓キーのどちらも押されていないとき
            fb = 0

        if keyboard.is_pressed("left"):  # ←キーが押されているとき
            if lr >= -30:
                lr -= 1
        elif keyboard.is_pressed("right"):  # →キーが押されているとき
            if lr <= 30:
                lr += 1
        else:  # ↑キーと↓キーのどちらも押されていないとき
            lr = 0

        if keyboard.is_pressed("d"):  # dキーが押されたとき、時計回りに45度回転
            tello.Send("cw 45")

        elif keyboard.is_pressed("a"):  # aキーが押されたとき、反時計回りに45度回転
            tello.Send("ccw 45")
        if keyboard.is_pressed("s"):  # sキーが押されたとき、写真を撮影
            takePhotoFlag = True

        if keyboard.is_pressed("0"):  # 映像モードの切り替え
            videoMode = 0
        elif keyboard.is_pressed("1"):  # 映像モード1
            videoMode = 1
        elif keyboard.is_pressed("2"):  # 映像モード2
            videoMode = 2
        elif keyboard.is_pressed("3"):  # 映像モード3
            videoMode = 3

        # ドローンを移動させる
        tello.Send(f"rc {lr} {fb} {ud} {0}", is_wait=False)


except KeyboardInterrupt:  # Ctrl+cキーが押されたとき
    tello.Send("land")
    flyingFlag = False  # 映像表示を終了させる
    exit()
