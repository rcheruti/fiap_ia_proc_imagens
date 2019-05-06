
import cv2
import numpy as np
import utils.utils as tt

# ----------------------------

# imagem =      cv2.imread('imagens/hellmanns-logo.jpg')
imagem =      cv2.imread('imagens/hellmanns-logo-vegan.png')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY )

# ----------------------------
# detector ORB
detectorORB =     cv2.ORB_create( 500 )
# para origem
keyPoints =       detectorORB.detect( imagemCinza, None )
keyPoints, desc = detectorORB.compute( imagemCinza, keyPoints )

# ----------------------------
# parâmetros para o FLANN
paramsIndex = {
  'algorithm':          6 ,
  'table_number':       6 ,
  'key_size':           12 ,
  'multi_probe_level':  1 ,
}
paramsBusca = {
  'checks':             50 ,
}
# comparador FLANN
comparadorFLANN = cv2.FlannBasedMatcher( paramsIndex, paramsBusca )

# ----------------------------
# ----------------------------
# resultado final

cap = cv2.VideoCapture('videos/hellmanns.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
delay = 10

while(cap.isOpened()):
  ret, frame = cap.read()
  if frame is None:
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  keyPointsAlvo = detectorORB.detect( gray, None )
  keyPointsAlvo, descAlvo = detectorORB.compute( gray, keyPointsAlvo )

  encontrados = comparadorFLANN.knnMatch( desc, descAlvo, k = 2 )

  matchesMask = [ [0,0] for i in range( len( encontrados ) ) ]

  try:
    for i, (m, n) in enumerate( encontrados ):
      if m.distance < 0.6 * n.distance:
        matchesMask[ i ] = [ 1, 0 ]
  except ValueError:
    pass

  draw_params = dict(
    matchColor = (0,255,0), 
    singlePointColor = (255,0,0), 
    matchesMask = matchesMask, 
    flags = 0 )

  image_detected = cv2.drawMatchesKnn(
      imagemCinza, 
      keyPoints, 
      gray, 
      keyPointsAlvo, 
      encontrados, 
      None, 
      **draw_params )

  cv2.imshow('Encontrar logo na embalagem', image_detected)
  if cv2.waitKey( delay ) & 0xFF == ord('q'):
    break


cap.release()
cv2.destroyAllWindows()
