# Esta função identifica os algarismos na foto, gera uma imagem quadrada
# de tamanho padrão com cada um deles centralizados e retorna os vetores que
# representam cada imagem

def separa_algarismos(filename):

    import numpy as np
    import cv2

    # deixa a imagem em escala de cinza e borra para reduzir o número de contornos
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)

    # binariza
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 135, 10)

    # identifica contornos na imagem
    kernel = np.ones((5, 5),np.uint8)
    bthresh = cv2.erode(thresh, kernel, iterations = 3)
    contours, hierarchy = cv2.findContours(bthresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # calcula as áreas dos retângulos construídos pelos contornos em contours
    # o retângulo é limitado entre x e x+h na horizontal e entre y e y+w na vertical
    area = np.zeros(len(contours))
    for i, ctr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(ctr)
        area[i] = w*h
    bigarea = np.amax(area[1:])

    # imaginando que os algarismos são os maiores contornos, é suposto que
    # todo contorno de tamanho superior a 15% do maior corresponde a um algarismo
    j = 1
    for i, ctr in enumerate(contours):
        x, y, w, h = cv2.boundingRect(ctr)
        if 0.15*bigarea < area[i] and hierarchy[0,i,3] == 0:
            cv2.imwrite("test_results/"+str(j)+".png",thresh[y:y+h,x:x+w])
            j += 1

    # adiciona borda branca
    Vetores = []

    for i in range(1,j):
        img = cv2.imread("test_results/"+str(i)+".png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m, n = (np.size(img,0),np.size(img,1))
        M = 255*np.ones((int(m+0.3*m), int(m + 0.3*m)))#, dtype = np.int8)
        M[int(0.15*m + 1): int(0.15*m + m) + 1, int((0.15*m + 0.5*m - 0.5*n) + 1): int((0.15*m + 0.5*m - 0.5*n) + n) + 1] = img

        # ajusta o traço das imagens, é preciso engordar os dígitos com razão m/n elevada
        kernel = np.ones((2, 2),np.uint8)
        if n < 0.9*m:
            M = cv2.erode(M, kernel, iterations = 1)
        M = cv2.erode(M, kernel, iterations = 1)

        M = cv2.resize(M,dsize = (28,28))
        Vetores.append(M.flatten())

    return Vetores
