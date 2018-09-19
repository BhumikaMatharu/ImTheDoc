from flask import Flask, jsonify, request
import base64
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from imutils import face_utils
import dlib
import os

app = Flask(__name__)

@app.route('/cloud1', methods=['POST'])
def get_tasks1():
    st1 = request.get_data()
    st = str(st1)
    #print st
    image_64_decode = base64.decodestring(st1)
    image_result = open('dtry.jpg', 'wb')
    image_result.write(st.decode('base64'))
    image_result.close()
    f = '/Users/shivamarora/Desktop/dtry.jpg'
    jaundice = False
    img_color = cv2.imread(f)
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
    detected = detector(img, 1)
        
    for i, detect in enumerate(detected):
        print 'processing...'
        shape = predictor(img, detect)
        shape = face_utils.shape_to_np(shape)
        
        for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == 'right_eye' or name == 'left_eye':
                print i,j, name
                x,y,w,h = cv2.boundingRect(np.array([shape[i:j]]))
                eye = img_color[y:y+h, x:x+w]
                
                #cv2.imshow("Eye", eye)
                #cv2.waitKey(0)
                cv2.imwrite(name+'.png', eye)
                lower = np.array([5,105,140])
                upper = np.array([50, 200, 260])
                mask = cv2.inRange(eye, lower, upper)
                output = cv2.bitwise_and(eye, eye, mask = mask)
                if np.max(output) != 0:
                    jaundice = True

    return 'Jaundice:'+str(jaundice)

if __name__ == '__main__':
    app.run(debug=True)