import numpy as np 
import cv2

VIDEO_PATH = r"C:\Users\vitor.matheus\Music\GIT - Corporativo\PAV\EmbarcadoCCR\channel1_20250.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

frames_Ids = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=120)

frames = []

for fid in frames_Ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
#print(f"Median frame calculated.", medianFrame)
cv2.imshow("Median Frame", medianFrame)
cv2.waitKey(0)

#cv2.imwrite("medianFrame.jpg", medianFrame)
