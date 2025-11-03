import cv2
import sys
import numpy as np 

VIDEO_PATH = r"C:\Users\vitor.matheus\Music\GIT - Pessoal\movement_detection\501-502 - Trim.mkv"

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

def subtractor(algorithm_type):
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = 120, 
                                                        decisionThreshold=0.8)
    if algorithm_type == 'MOG':
        return cv2.bgsegm.createBackgroundSubtractorMOG(history = 100, nmixtures = 5,
                                                        backgroundRatio = 0.7, 
                                                        noiseSigma = 0)
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2(history = 500, detectShadows=True,
                                                varThreshold=100)
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, 
                                                 detectShadows=True)
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, 
                                                        useHistory =True,
                                                        maxPixelStability=15*60,
                                                        isParallel=True)
    print('Detector inválido')
    sys.exit(1)


w_min = 50 # Largura minima do retângulo
h_min = 50 # Altura minima do retângulo
offset = 3 # Erro permitido entre o centro do objeto e a linha
linha_roi = 250 # Posição da linha de contagem
carros = 0

def centroide(x, y, w, h):
    """
    :param x: x do objeto
    :param y: y do objeto
    :param w: largura do objeto
    :param h: altura do objeto
    :return: tupla com as coordenadas do centroide
    """
    x1 = w // 2
    y1 = h // 2
    cx = x + x1
    cy = y + y1
    return cx, cy   

detec = []
def set_info(detec):
    global carros
    for (x, y) in detec:
        if (linha_roi + offset) > y > (linha_roi - offset):
            carros += 1
            cv2.line(frame, (25, linha_roi), (300, linha_roi), (0, 127, 255), 3)
            detec.remove((x, y))
            print("Carros detectados até o momento: " + str(carros))

def show_info(frame, mask):
    text = f'Carros: {carros}'
    cv2.putText(frame, text, (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Original", frame)
    cv2.imshow("Detectar", mask)

cap = cv2.VideoCapture(VIDEO_PATH)
background_substractor = subtractor(algorithm_type)

e1 = cv2.getTickCount()

while True:    
    ok, frame = cap.read()

    if not ok:
        print("Frames acabaram")
        break

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)    

    mask = background_substractor.apply(frame)
    mask = Filter(mask, 'combine') 
    #car_after_mask = cv2.bitwise_and(frame, frame, mask=mask)


    contorno, img = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (25, linha_roi), (300, linha_roi), (255, 127, 0), 3)
    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= w_min) and (h >= h_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centro = centroide(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame, centro, 4, (0, 0, 255), -1)

    set_info(detec)
    show_info(frame, mask)


    if cv2.waitKey(1) == 27: #ESC
        break

cv2.destroyAllWindows()
cap.release()