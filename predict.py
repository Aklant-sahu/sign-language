import cv2
import mediapipe as mp
import time
import pandas as pd
import pickle
from sklearn.externals import joblib
import numpy as np


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

#tragets=['A','B','C']

y=[]
# feat_dict={'1x':[],'1y':[],'2x':[],'2y':[],'3x':[],'3y':[],'4x':[],'2y':[]}
feat_dict = dict()
for i in range(0,21):
    feat_dict[f'{i}x'] = []
    feat_dict[f'{i}y'] = []

while True:
    success, img = cap.read()
    #print(img.shape)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    
    pos=[]
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            
            for id, lm in enumerate(handLms.landmark):
                
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                pos.append([f'{id}x',cx])
                pos.append([f'{id}y',cy])
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.rectangle(img,(50,110),(300,390),color=(0,0,255),thickness=2)
    cv2.imshow("Image", img)
    
    k=cv2.waitKey(1)
    if k==ord('c'):
        #print(pos)
        for i in range(len(pos)):
            feat_dict[pos[i][0]].append(pos[i][1])
            

        df=pd.DataFrame(feat_dict)
        filename = 'A-B-C-D-E-F-G-H-I-J-K-L-U-V-W-Y'
        base_mod= pickle.load(open(filename, 'rb'))
        scaler_filename = "scaler.save"
        scale= joblib.load(scaler_filename) 
        data=df.iloc[0].values
        print(data)
        data=scale.transform(np.array(data).reshape(-1,data.shape[0]))
        pred=base_mod.predict(data)
        print(pred)
            







            

        #print(feat_dict)
        

        

    elif k==ord('q'):
        cv2.destroyAllWindows()
        # df=pd.DataFrame(feat_dict)
        # df.to_csv('B.csv')
        
        #print(pos[:10])
        break