import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier( 'cascades/data/haarcascade_frontalface_alt2.xml' )
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read( "trainer.yml" )

labels = {}
with open( "labels.pickle", 'rb' ) as f :
    org_labels = pickle.load( f )
    labels = { v : k for k, v in org_labels.items() }

cap = cv2.VideoCapture( 0 )

while True :
    ret, frame = cap.read()

    gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
    faces = face_cascade.detectMultiScale( gray, scaleFactor = 1.5, minNeighbors = 5 )

    for ( x, y, w, h ) in faces :
        roi_gray = gray[ y : y + h, x : x + w ]
        roi_color = frame[ y : y + h, x : x + w ]

        id_, _loss = recognizer.predict( roi_gray )
        print( id_, _loss )
        # print( labels[ id_ ] )

        if _loss < 60 :
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[ id_ ]
            color = ( 255, 255, 255 )
            stroke = 2
            cv2.putText( frame, name, ( x, y ), font, 1, color, stroke, cv2.LINE_AA )

            color = ( 0, 255, 0 )
            stroke = 2
            cv2.rectangle( frame, ( x, y ), ( x + w, y + h ), color, stroke )
        else :
            color = ( 0, 0, 255 )
            stroke = 2
            cv2.rectangle( frame, ( x, y ), ( x + w, y + h ), color, stroke )

        # img_item = "temp.jpg"
        # cv2.imwrite( img_item, roi_color )

        eyes = eye_cascade.detectMultiScale( roi_gray )
        for ( ex, ey, ew, eh ) in eyes:
            cv2.rectangle( roi_color, ( ex, ey ),( ex + ew, ey + eh ),( 255, 0, 0), 2 )

    if len(faces) > 0:
        cv2.putText(frame, "Number of faces detected: " + str(faces.shape[0]), (0,frame.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,255,0), 1)
    else:
        cv2.putText(frame, "No face detected ", (0,frame.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,255), 1)
    
    # cv2.imshow( "gray", gray )
    # cv2.imshow( "frame1", frame )
    # cv2.imshow( "frame2", frame )
    cv2.imshow( 'frame', frame )

    if ( cv2.waitKey( 20 ) & 0xFF ) == ord( 'q' ) :
        break

cap.release()
cv2.destroyAllWindows()
