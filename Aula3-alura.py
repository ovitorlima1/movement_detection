import cv2
import sys

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
background_substractor = substractor(algorithm_type)

e1 = cv2.getTickCount()
def main():
    frame_number = -1 
    while (cap.isOpened()):
    
        ok, frame = cap.read()

        if not ok:
            print("Frames acabaram")
            break
    
        frame_number += 1
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)    
        mask = background_substractor.apply(frame)
        cv2.imshow("Frame", frame) 
        cv2.imshow("Mask", mask)
        if cv2.waitKey(30) & 0xFF == ord('c') or frame_number == 300:
            break
    e2 = cv2.getTickCount()
    time = (e2 - e1)/cv2.getTickFrequency()
    print("Tempo de execução: ", time)
    cap.release()
    cv2.destroyAllWindows()
main()