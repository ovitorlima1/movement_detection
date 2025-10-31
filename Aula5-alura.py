import cv2
import sys
import numpy as np 

VIDEO_PATH = r"C:\Users\vitor.matheus\Music\GIT - Pessoal\movement_detection\channel1_20250.mp4"

algorithm_types = ['KNN', 'GMG', 'CNT', 'MOG', 'MOG2']
algorithm_type = algorithm_types[3]

def Kernel(KERNEL_TYPE):
    if KERNEL_TYPE == 'dilation':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    if KERNEL_TYPE == 'opening':
        kernel = np.ones((15,15), np.uint8)
    if KERNEL_TYPE == 'closing':
        kernel = np.ones((15,15), np.uint8)

    return kernel

def Filter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
    if filter == 'dilation':
        return cv2.dilate(img, Kernel('dilation'), iterations=2)
    
    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, Kernel('closing'), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, Kernel('opening'), iterations=2)
        dilation = cv2.dilate(opening, Kernel('dilation'), iterations=2)
        return dilation
    

print("Dilation Kernel: ")
print(Kernel('dilation'))

print("Opening Kernel: ")
print(Kernel('opening'))

print("Dilation Kernel: ")
print(Kernel('closing'))


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
        mask_Filter = Filter(mask, 'closing') 
        car_after_mask = cv2.bitwise_and(frame, frame, mask=mask_Filter)
        

        cv2.imshow("Frame", frame) 
        cv2.imshow("Mask", mask)
        cv2.imshow("Mask Filtered", mask_Filter)
        cv2.imshow("Car After Mask", car_after_mask)    
        if cv2.waitKey(30) & 0xFF == ord('c') or frame_number == 600:
            break


    e2 = cv2.getTickCount()
    time = (e2 - e1)/cv2.getTickFrequency()
    print("Tempo de execução: ", time)
    cap.release()
    cv2.destroyAllWindows()
main()