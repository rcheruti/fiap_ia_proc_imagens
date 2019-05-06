
import cv2
import numpy as np
import utils.utils as tt
# print(cv2.__version__)

# ----------------------------

imagem =      cv2.imread('imagens/mario.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY )
template =    cv2.imread('imagens/coin_mario.jpg', cv2.IMREAD_GRAYSCALE )
w, h = template.shape[ ::-1 ]

copiaImg = imagem.copy()
encontradas = cv2.matchTemplate( imagemCinza, template, cv2.TM_CCOEFF_NORMED )

proximidade = 0.9 # porcentagem de proximidade do "template" com alguma parte da imagem
localizacao = np.where( encontradas > proximidade )

# print( localizacao )
# cacheX = {}
# cacheY = {}
itensEncontrados = 0
itensPulados = 0
for ponto in zip( *localizacao[ ::-1 ] ):
  # if( ponto[0] in cacheX or ponto[1] in cacheY ):
  #   itensPulados += 1
  #   continue
  # cacheX[ ponto[0] ] = True
  # cacheY[ ponto[1] ] = True
  itensEncontrados += 1
  cv2.rectangle( copiaImg, ponto, ( ponto[0] + w, ponto[1] + h ), (0, 255, 0), 1 )
  pass

print('Encontrados: %d, Pulados: %d' % ( itensEncontrados, itensPulados ))
tt.mostrarImagem( copiaImg )
