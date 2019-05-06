
import cv2
import numpy as np
import utils.utils as tt

# ----------------------------

cap = cv2.VideoCapture('videos/hellmanns.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
print('FPS: %d, Delay: %dms' % ( fps, delay ))
qtdFrames = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    qtdFrames += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey( delay ) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Quantidade de frames: %d' % ( qtdFrames ))
