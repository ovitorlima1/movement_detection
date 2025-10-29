import cv2
import sys
import csv
import numpy as np

fp = open('results.csv', mode='w')
writer = csv.DictWriter(fp, fieldnames=["Frame", "Pixel count"])
writer.writeheader()



VIDEO_PATH = r"C:\Users\vitor.matheus\Music\GIT - Pessoal\movement_detection\channel1_20250.mp4"

algorithm_types = ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2']
algorithm_type = algorithm_types[4]

## KNN = Tempo de execução:  14.332573
## GMG = Tempo de execução:  14.6685902
## CNT = Tempo de execução:  14.155815
## MOG = Tempo de execução:  14.2354167
## MOG2 = Tempo de execução: 14.2768369


def substractor(algorithm_type):

    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    
    print("ERRO - Insira uma nova informação")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
background_substractor = []

for i, a in enumerate(algorithm_types):
    print(f"Creating substractor for {a}")
    background_substractor.append(substractor(a))


#e1 = cv2.getTickCount()
def main():
    frame_number = -1 
    while (cap.isOpened()):
    
        ok, frame = cap.read()

        if not ok:
            print("Frames acabaram")
            break
    
        frame_number += 1
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)    
        
        knn = background_substractor[0].apply(frame)
        gmg = background_substractor[1].apply(frame)
        cnt = background_substractor[2].apply(frame)
        mog = background_substractor[3].apply(frame)
        mog2 = background_substractor[4].apply(frame)

        knnCount = cv2.countNonZero(knn)
        gmgCount = cv2.countNonZero(gmg)
        cntCount = cv2.countNonZero(cnt)
        mogCount = cv2.countNonZero(mog)
        mog2Count = cv2.countNonZero(mog2)

        writer.writerow({"Frame": "KNN", "Pixel count": knnCount })
        writer.writerow({"Frame": "GMG", "Pixel count": gmgCount})
        writer.writerow({"Frame": "CNT", "Pixel count": cntCount})
        writer.writerow({"Frame": "MOG", "Pixel count": mogCount})
        writer.writerow({"Frame": "MOG2", "Pixel count": mog2Count})
        

        cv2.imshow("Frame", frame) 
        cv2.imshow("KNN", knn)
        cv2.imshow("GMG", gmg)
        cv2.imshow("CNT", cnt)
        cv2.imshow("MOG", mog)
        cv2.imshow("MOG2", mog2)
        if cv2.waitKey(30) & 0xFF == ord('c') or frame_number == 300:
            break
    #e2 = cv2.getTickCount()
    #time = (e2 - e1)/cv2.getTickFrequency()
    #print("Tempo de execução: ", time)
    cap.release()
    cv2.destroyAllWindows()
main()

