
import cv2

# ------------------------------

def mostrarImagem( cv2Imagem, titulo = '' ):
  cv2.imshow(titulo, cv2Imagem)
  cv2.waitKey()
  cv2.destroyAllWindows()
  pass
