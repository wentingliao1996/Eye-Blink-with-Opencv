import cv2
import cvzone
from  cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture('/eye_contect/video/smaple.mp4')
detector = FaceMeshDetector(maxFaces=1) #辨識臉部468個點位,maxFaces指辨識一張臉
plotY = LivePlot(640, 360, [25,50], invert=True)

ratioList = []
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
blinkCounter = 0
counter = 0
color = (255, 0, 255)

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()
    print(success) 
    img , faces = detector.findFaceMesh(img, draw= False)#將影像套用至detector

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)
        
        leftUP = face[159]
        leftDown = face[23]
        leftRight = face[130]
        leftLeft = face[243]
        lengthVer, _ = detector.findDistance(leftUP, leftDown)
        lengthHor, _ = detector.findDistance(leftRight, leftLeft)

        cv2.line(img, leftUP, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftRight, leftLeft, (0, 200, 0), 3)

        ratio = (lengthVer/lengthHor)*100
        ratioList.append(ratio)
        if len(ratioList)>5:#以10次為1單位
            ratioList.pop(0)#超過10就從最後一個單位在開始計算
        ratioAvg = sum(ratioList)/len(ratioList)

        if ratioAvg< 30 and counter ==0:
            blinkCounter +=1
            counter = 1
            color = (0,200, 0)
        if counter != 0:
            counter +=1
            if counter >10:
                counter = 0
                color = (255, 0,255)
        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (100,100), colorR=color)
        
        imgPlot = plotY.update(ratioAvg, color)
        img = cv2.resize(img, (640,360))#調整輸出影像大小以契合兩個影像
        ImgStack = cvzone.stackImages([img, imgPlot], 2, 1)#合併兩個影像(要相似大小才能合併)
    else:#若無發現臉 則顯示兩個一樣的影像
        img = cv2.resize(img, (640,360))
        ImgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow('Image', ImgStack)
    cv2.waitKey(25)
