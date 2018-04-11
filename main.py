import  cv2
import numpy as np

camera = cv2.VideoCapture('1.mp4')

es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
"""返回指定形状和尺寸的的结构元素，矩形：MORPH_RECT，交叉形：MORPH_CORSS，椭圆形：MORPH_ELLIPSE"""
kernel = np.ones((5,5),np.uint8)
background = None

while(True):
    ret, frame = camera.read()
    if background is None:
        background = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#将每一帧图像由RGB转换成灰度图
        background = cv2.GaussianBlur(background,(21, 21),0)
        continue
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame,(21,21),0)

    diff = cv2.absdiff(background, gray_frame)#两幅图像做差
    diff = cv2.threshold(diff, 25 ,255,cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, es, iterations = 2)

    image, cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c)<1500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y),(x + w, y + h),(0, 255, 0),2)

    cv2.imshow("contours", frame)
    cv2.imshow("dif", diff)

    if cv2.waitKey(int(1000 / 12)) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
camera.release()



