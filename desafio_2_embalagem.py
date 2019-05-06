
import cv2
import numpy as np
import utils.utils as tt

# ----------------------------

# imagem =      cv2.imread('imagens/hellmanns-logo.jpg')
imagem =      cv2.imread('imagens/hellmanns-logo-vegan.png')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY )
copiaImg =    imagem.copy()
imagemAlvo =  cv2.imread('imagens/hellmanns-capture.png')
imagemAlvoCinza = cv2.cvtColor(imagemAlvo, cv2.COLOR_BGR2GRAY )

# ----------------------------
# detector ORB
detectorORB =     cv2.ORB_create( 500 )
# para origem
keyPoints =       detectorORB.detect( imagemCinza, None )
keyPoints, desc = detectorORB.compute( imagemCinza, keyPoints )
# para alvo
keyPointsAlvo =   detectorORB.detect( imagemAlvoCinza, None )
keyPointsAlvo, descAlvo = detectorORB.compute( imagemAlvoCinza, keyPointsAlvo )

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
encontrados =     comparadorFLANN.knnMatch( desc, descAlvo, k = 2 )


# ----------------------------
# ----------------------------
# resultado final

matchesMask = [ [0,0] for i in range( len( encontrados ) ) ]

for i, (m, n) in enumerate( encontrados ):
  if m.distance < 0.7 * n.distance:
    matchesMask[ i ] = [ 1, 0 ]

draw_params = dict(
  matchColor = (0,255,0), 
  singlePointColor = (255,0,0), 
  matchesMask = matchesMask, 
  flags = 0 )

image_detected = cv2.drawMatchesKnn(
    imagemCinza, 
    keyPoints, 
    imagemAlvoCinza, 
    keyPointsAlvo, 
    encontrados, 
    None, 
    **draw_params )

tt.mostrarImagem( image_detected )
