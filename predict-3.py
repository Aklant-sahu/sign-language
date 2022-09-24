import cv2
import mediapipe as mp
import time
import pandas as pd
import pickle
from sklearn.externals import joblib
import numpy as np
from pprint import pprint

feat_dict = dict()
for i in range(0,21):
    feat_dict[f'{i}x'] = []
    feat_dict[f'{i}y'] = []

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

y=[]
j=1


ans = str()
while True:
    
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
   
    pos=[]
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
           
            for id, lm in enumerate(handLms.landmark):
               
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                pos.append([f'{id}x',cx])
                pos.append([f'{id}y',cy])
                
                cv2.circle(img,(cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
     
    k=cv2.waitKey(1)
    if k==ord('c'):
        
            
                
        #print(pos)
        for i in range(len(pos)):
            feat_dict[pos[i][0]].append(pos[i][j])
        
        j+=1   
        df=pd.DataFrame(feat_dict)
        
        # Load model
        filename = 'A-B-C-D-E-F-G-H-I-J-K-L-U-V-W-Y'
        base_mod= pickle.load(open(filename, 'rb'))
        scaler_filename = "scaler.save"
        scale= joblib.load(scaler_filename)
        data=df.iloc[0].values
        data=scale.transform(np.array(data).reshape(-1,data.shape[0]))
        pred=base_mod.predict(data)

        string = np.array_str(pred)
        ans =  string[2]

    elif k==ord('q'):
        cv2.destroyAllWindows()
        break

    cv2.putText(img,str(ans), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
    cv2.rectangle(img,(50,110),(300,390),color=(0,0,255),thickness=2)
    cv2.imshow("Image", img)