
import cv2
import numpy as np
import utils.utils as tt

# ----------------------------

imagem =      cv2.imread('imagens/cristo.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY )
copiaImg =    imagem.copy()

# Estes códigos abaixo serão executados pelo algoritmo ORB
# detectorFAST = cv2.FastFeatureDetector_create()
# detectorBRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# keyPoints = detectorFAST.detect( imagemCinza, None )
# keyPoints, desc = detectorBRIEF.compute( imagemCinza, keyPoints )

detectorORB = cv2.ORB_create( 500 )
keyPoints = detectorORB.detect( imagemCinza, None )
keyPoints, desc = detectorORB.compute( imagemCinza, keyPoints )

# print('Pontos detectados: %d' % ( len(keyPoints) ))
# copiaImg = cv2.drawKeypoints( copiaImg, keyPoints, copiaImg,
#   flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
# tt.mostrarImagem( copiaImg, 'Cristo' )

# ---------------------------

imagemAlvo =  cv2.imread('imagens/cristo-redentor.jpg')
imagemAlvoCinza = cv2.cvtColor(imagemAlvo, cv2.COLOR_BGR2GRAY )

keyPointsAlvo = detectorORB.detect( imagemAlvoCinza, None )
keyPointsAlvo, descAlvo = detectorORB.compute( imagemAlvoCinza, keyPointsAlvo )

# FLANN_INDEX_LSH = 6
paramsIndex = {
  'algorithm':          6 ,
  'table_number':       6 ,
  'key_size':           12 ,
  'multi_probe_level':  1 ,
}
paramsBusca = {
  'checks':             50 ,
}

comparadorFLANN = cv2.FlannBasedMatcher( paramsIndex, paramsBusca )
encontrados = comparadorFLANN.knnMatch( desc, descAlvo, k = 2 )

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
