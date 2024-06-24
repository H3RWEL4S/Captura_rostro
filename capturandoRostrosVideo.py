import cv2 #Manipulación de imágenes y video
import os  #Creación de directorios

#1 Creación de carpeta para almacenamiento de rostros encontrados
if not os.path.exists('Rostros encontrados'):
	print('Carpeta creada: Rostros encontrados')
	os.makedirs('Rostros encontrados')

#2 Captura de video forzando el modo DirectShow de Windows
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)#Uso cv2.CAP_DSHOW, al momento de  realizar la visualización ocupe totalmente la ventana

#3 Clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

count = 0

#4 Procesamiento de fotogramas de la camara
while True:
	ret,frame = cap.read() #captura de fotograma
	frame = cv2.flip(frame,1) #reflejo horizontal del fotograma capturado
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #conversión a escala de grises
	auxFrame = frame.copy() #copia del fotograma original
#4-1 Detección de rotors
	faces = faceClassif.detectMultiScale(gray, 1.3, 5)
#4-2 Manejo de tecla de escape
	k = cv2.waitKey(1)
	if k == 27:
		break
#4-3 Dibujo de rectangulos alrededor de los rostros
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(128,0,255),2)
		#4-3-1 Extraccion y direccionamiento del rostro
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)
		#4-3-2 Almacenamiento del rostro
		if k == ord('s'):
			cv2.imwrite('Rostros encontrados/rostro_{}.jpg'.format(count),rostro)
			cv2.imshow('rostro',rostro)
			count = count +1
	#4-4 Superposicion de texto		
	cv2.rectangle(frame,(10,5),(450,25),(255,255,255),-1)
	cv2.putText(frame,'Presione S, para almacenar los rostros encontrados',(10,20), 2, 0.5,(128,0,255),1,cv2.LINE_AA)
	cv2.imshow('frame',frame)
#5 StopRun
cap.release()
cv2.destroyAllWindows()