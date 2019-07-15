import cv2
import matplotlib.pyplot as plt
import numpy as np

def extracao(filename, linmax):
    Vet = []
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 135, 15)
    thresh = cv2.resize(thresh, dsize = (1520,2280), interpolation=cv2.INTER_AREA)


    for i in range(linmax):
        for k in range(10):

            crop_img = (thresh[12+i*152:(i+1)*152-12, 12+k*152:(k+1)*152-12])

            # identifica contornos na imagem
            kernel = np.ones((5, 5),np.uint8)
            bcrop_img = cv2.erode(crop_img, kernel, iterations = 1)
            contours, hierarchy = cv2.findContours(bcrop_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            area = np.zeros(len(contours))
            for j, ctr in enumerate(contours):
                x, y, w, h = cv2.boundingRect(ctr)
                area[j] = w*h
            if len(area) > 1:
                bigarea = np.amax(area[0:-1])
            else:
                bigarea = np.amax(area[0])

            for j, ctr in enumerate(contours):
                x, y, w, h = cv2.boundingRect(ctr)
                if bigarea == area[j]:
                    crop_img = crop_img[y:y+h,x:x+w]

            m, n = (np.size(crop_img,0),np.size(crop_img,1))
            if m >= n:
                M = 255*np.ones((int(m+0.36*m), int(m + 0.36*m)))#, dtype = np.int8)
                M[round(0.18*m + 1): round(0.18*m + m) + 1, round((0.18*m + 0.5*m - 0.5*n) + 1): round((0.18*m + 0.5*m - 0.5*n) + n) + 1] = crop_img
            else:
                M = 255*np.ones((int(n+0.36*n), int(n + 0.36*n)))#, dtype = np.int8)
                M[round(0.18*n + 1): round(0.18*n + m) + 1, round((0.18*n + 0.5*m - 0.5*n) + 1): round((0.18*n + 0.5*m - 0.5*n) + n) + 1] = crop_img

            crop_img = cv2.resize(M,dsize=(28,28),interpolation=cv2.INTER_AREA)

            # padrao de entrada para o classificador
            crop_img = np.reshape(crop_img, (784,))
            Vet.append(crop_img)
    return(Vet)

# para salvar banco de dados convertido

#for linmax in range(1,16,1):

#    Vet = []
#    Target = []

#    for i in range(15):#para cada folha
#        for k in range(linmax):
#            for j in range(10):
#                Target.append(j)
#        Vet = Vet + extracao('folha%i.png'%(i+1), linmax)

#        n_dados, _ = np.shape(Vet)

#    np.save('Saidas/Num' + str(n_dados) + '.npy',Vet)
#    np.save('Saidas/Target' + str(n_dados) + '.npy',Target)
