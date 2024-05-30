import cv2

from cvzone.FaceDetectionModule import FaceDetector

video = cv2.VideoCapture(cv2.CAP_DSHOW) # para rodar o video 'hinobrasil.mp4' e para rodar a webcam 0, cv2.CAP_DSHOW

detector = FaceDetector(minDetectionCon=0.5)

while True:

    ret, img = video.read()

    if not ret:
        print("Erro ao capturar imagem da câmera.")
        break

    img, bboxes = detector.findFaces(img, draw=False)
    img2 = img.copy()

    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox['bbox']

            rec = img[y:y + h, x:x + w]

            if rec.size != 0:
                try:
                    recBlur = cv2.blur(rec, (30, 30))
                    img2[y:y + h, x:x + w] = recBlur
                except cv2.error as e:
                    print(f"Erro ao aplicar o desfoque: {e}")
            else:
                print("Programa não está identificando rosto.")


    cv2.imshow('Imagem com rosto normal', img)

    cv2.imshow('Imagem com rosto borrado', img2)


    if cv2.waitKey(1) & 0xFF == ord('e'):
        break


video.release()

cv2.destroyAllWindows()

