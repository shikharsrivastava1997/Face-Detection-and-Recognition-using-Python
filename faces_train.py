import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )
image_dir = os.path.join( BASE_DIR, "images" )

face_cascade = cv2.CascadeClassifier( 'cascades/data/haarcascade_frontalface_alt2.xml' )
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}

x_train = []
y_labels = []

for root, dirs, files in os.walk( image_dir ) :
    for file in files :
        if file.endswith("JPG") or file.endswith("jpg") or file.endswith("jpeg") or file.endswith("JPEG"):
            path = os.path.join( root,  file )
            label = os.path.basename( root )

            # print( path, label )

            if not label in label_ids :
                label_ids[ label ] = current_id
                current_id += 1

            id_ = label_ids[ label ]

            # print( label_ids )

            pil_image = Image.open( path ).convert( "L" )
            if current_id == 1 :
                image_array = np.array( pil_image, "uint8" )
            elif current_id == 2 :
                size = ( 900, 900 )
                final_image = pil_image.resize( size, Image.ANTIALIAS )
                image_array = np.array( final_image, "uint8" )
            elif current_id == 3 :
                size = ( 1000, 1000 )
                final_image = pil_image.resize( size, Image.ANTIALIAS )
                image_array = np.array( final_image, "uint8" )
            else :
                size = ( 700, 1100 )
                final_image = pil_image.resize( size, Image.ANTIALIAS )
                image_array = np.array( final_image, "uint8" )

            # image_array = np.array( pil_image, "uint8" )

            faces = face_cascade.detectMultiScale( image_array, scaleFactor = 1.5, minNeighbors = 5 )

            for ( x, y, w, h ) in faces :
                roi = image_array[ y : y + h, x : x + w ]
                x_train.append( roi )
                y_labels.append( id_ )

with open( "labels.pickle", 'wb' ) as f :
    pickle.dump( label_ids, f )

recognizer.train( x_train, np.array( y_labels ) )
recognizer.save( "trainer.yml" )
