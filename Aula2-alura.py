import numpy as np 
import cv2
from time import sleep


VIDEO_PATH = r"C:\Users\vitor.matheus\Music\GIT - Pessoal\movement_detection\channel1_20250.mp4"
DELAY = 10


cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

frames_Ids = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=240)

frames = []

for fid in frames_Ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
#print(f"Median frame calculated.", medianFrame)
#cv2.imshow("Median Frame", medianFrame)
#cv2.waitKey(0)

#cv2.imwrite("medianFrame.jpg", medianFrame)

# Aula 2 - Convertendo para escala de cinza
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Gray Median Frame", grayMedianFrame)
#cv2.waitKey(0)


while True:

    tempo = float(1/DELAY)
    sleep(tempo)

    ret, frame = cap.read()

    if not ret:
        print("Acabou os frames.")
        break

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dframe = cv2.absdiff(frameGray, grayMedianFrame)
    th, dframe = cv2.threshold(dframe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow("Frame Gray", dframe)


    if cv2.waitKey(1) & 0xFF == ord('c'):
        break
cap.release()
