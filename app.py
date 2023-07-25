from flask import Flask,redirect,url_for,render_template,request,Response
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS  
from playsound import playsound 
import os
language="en"

app=Flask(__name__)

cap = cv2.VideoCapture(0)

def gen_biceps():
       mp_drawing = mp.solutions.drawing_utils
       mp_pose = mp.solutions.pose
       
       def calculate_angle(a,b,c):
              a = np.array(a) # First
              b = np.array(b) # Mid
              c = np.array(c) # End
        
              radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
              angle = np.abs(radians*180.0/np.pi)
        
              if angle >180.0:
                     angle = 360-angle
            
              return angle
       # Curl counter variables
       counter1 = 0 
       counter2=0
       stage1 = None
       stage2 = None

       c1=-1
       c2=-1
       ## Setup mediapipe instance
       with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
              while cap.isOpened():
                     ret, frame = cap.read()
                            
                     # Recolor image to RGB
                     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     image.flags.writeable = False
                     
                     # Make detection
                     results = pose.process(image)
                     
                     # Recolor back to BGR
                     image.flags.writeable = True
                     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            
                     # Extract landmarks
                     try:
                            landmarks = results.pose_landmarks.landmark
                                   
                            # Get coordinates
                            shoulder1 = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            elbow1 = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                            wrist1 = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                                   
                            # Calculate angle
                            angle1 = calculate_angle(shoulder1, elbow1, wrist1)
                                   
                            # Visualize angle
                            cv2.putText(image, str(angle1), 
                                          tuple(np.multiply(elbow1, [640, 480]).astype(int)), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                 )
                                   
                            shoulder2 = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                            elbow2 = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                            wrist2 = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                                   
                            # Calculate angle
                            angle2 = calculate_angle(shoulder2, elbow2 , wrist2)
                                   
                            # Visualize angle
                            cv2.putText(image, str(angle2), 
                                          tuple(np.multiply(elbow2, [640, 480]).astype(int)), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                 )
                            # Curl counter logic
                            if angle1 > 150:
                                   stage1 = "down"
                                   if angle1 > 165:
                                          stage1="wrong";
                            if angle1 < 50 and stage1 =='down':
                                   stage1="up"
                                   counter1 +=1
                                   print("right:",end=" ")
                                   print(counter1)
                                          
                            if angle2 > 150:
                                   stage2 = "down"
                                   if angle2 > 165:
                                          stage2="wrong";
                            if angle2 < 50 and stage2 =='down':
                                   stage2="up"
                                   counter2 +=1
                                   print("left:",end=" ")
                                   print(counter2)
                                          
                     except:
                            pass
                     if c1 != counter1:
                            txt = "Right" + str(counter1)
                            counter = gTTS(text=txt, lang=language, slow=False)   
                            counter.save("rcounter.mp3")  
                            playsound("rcounter.mp3")
                            os.remove("rcounter.mp3")
                     if c2 != counter2:
                            txt = "Left" + str(counter2)
                            counter = gTTS(text=txt, lang=language, slow=False)   
                            counter.save("lcounter.mp3")  
                            playsound("lcounter.mp3")
                            os.remove("lcounter.mp3")
                     c1 = counter1
                     c2 = counter2
                            
                     # Render curl counter
                     # Setup status box
                     cv2.rectangle(image, (0,0), (170,90), (0,0,0), -1)

                     #rep data right
                     cv2.putText(image,'LEFT',(50,30),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(16,255,255),2,cv2.LINE_AA)
                     cv2.putText(image, 'REPS', (5,50), 
                            cv2.FONT_HERSHEY_SIMPLEX,0.6 , (255,255,255), 1, cv2.LINE_AA)
                     cv2.putText(image, str(counter2), 
                            (10,85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                            
                     # Stage data
                     cv2.putText(image, 'STAGE', (100,50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                     if stage2=="wrong":
                            cv2.putText(image, stage2, 
                            (78,85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                     else:
                            cv2.putText(image, stage2, 
                            (78,85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                            
                     cv2.rectangle(image, (469,0), (639,90), (0,0,0), -1)

                     cv2.putText(image,'RIGHT',(519,30),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(16,255,255),2,cv2.LINE_AA)
                     # Rep data
                     cv2.putText(image, 'REPS', (474,50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,255,255), 1, cv2.LINE_AA)
                     cv2.putText(image, str(counter1), 
                            (479,85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                            
                     # Stage data
                     cv2.putText(image, 'STAGE', (569,50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 , (255,255,255), 1, cv2.LINE_AA)
                     #cv2.putText(image, stage1, 
                            #(480,85), 
                            #cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                     if stage1=="wrong":
                            cv2.putText(image, stage1, 
                            (547,85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                     else:
                            cv2.putText(image, stage1, 
                            (547,85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                            
                            
                     cv2.rectangle(image,(0,120),(270,160),(0,0,0),1)
                     #Advice-1
                     if stage2=="up":
                            cv2.putText(image,'Advice:lowerdown',(3,150),
                                   cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA
                                   )
                     else:
                            cv2.putText(image,'Advice:raiseup',(3,150),
                                   cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA
                                   )
                     #Advice-2
                     cv2.rectangle(image,(369,120),(639,160),(0,0,0),1)
                     if stage1=="up":
                            cv2.putText(image,'Advice:lowerdown',(372,150),
                                   cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA
                                   )
                     else:
                            cv2.putText(image,'Advice:raiseup',(372,150),
                                   cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA
                                   )
                            
                     # Render detections
                     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                          )               
                            
                     cv2.imshow('Bicep curls', image)

                     if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

              cap.release()
              cv2.destroyAllWindows()

def gen_shoulder():
       mp_drawing = mp.solutions.drawing_utils
       mp_pose = mp.solutions.pose
       def calculate_angle(a,b,c):
              a = np.array(a) # First
              b = np.array(b) # Mid
              c = np.array(c) # End
        
              radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
              angle = np.abs(radians*180.0/np.pi)
        
              if angle >180.0:
                     angle = 360-angle
            
              return angle
       
       # Curl counter variables
       counter = 0 
       stage = None

       ## Setup mediapipe instance
       with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
              while cap.isOpened():
                     ret, frame = cap.read()
                     
                     # Recolor image to RGB
                     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     image.flags.writeable = False
              
                     # Make detection
                     results = pose.process(image)
              
                     # Recolor back to BGR
                     image.flags.writeable = True
                     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                     
                     # Extract landmarks
                     try:
                            landmarks = results.pose_landmarks.landmark
                            
                            # Get coordinates
                            Hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                            
                            
                            # Calculate angle
                            angle = calculate_angle(Hip, shoulder, elbow)
                            
                            # Visualize angle
                            cv2.putText(image, str(angle), 
                                          tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                 )
                            
                            # shoulder press logic
                            if angle < 90:
                                   stage = "down"
                                   if angle>40 and angle<50:
                                          stage="CORRECT";
                                   elif angle<40:
                                          stage="Wrong"
                            if angle >140 and stage =='down':
                                   stage="up"
                                   counter +=1
                                   print(counter)
                                   
                     except:
                            pass
                     
                     # Render curl counter
                     # Setup status box
                     cv2.rectangle(image, (0,0), (215,90), (0,0,0), -1)
                     
                     # Rep data
                     cv2.putText(image, 'REPS', (1,40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                     cv2.putText(image, str(counter), 
                            (5,85), 
                            cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2, cv2.LINE_AA)
                     
                     # Stage data
                     cv2.putText(image, 'STAGE', (90,40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                     
                     if stage=='Wrong':
                            cv2.putText(image, stage, 
                            (78,85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                     else:
                            cv2.putText(image, stage, 
                            (78,85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                     cv2.rectangle(image,(400,0),(639,60),(0,0,0),1)
                     cv2.putText(image,'Advice:Raiseup ',(405,50),
                                   cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA
                                   )
                     
                     # Render detections
                     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                          )               
                     
                     cv2.imshow('Mediapipe Feed', image)

                     if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

              cap.release()
              cv2.destroyAllWindows()

def gen_triceps():
       mp_drawing = mp.solutions.drawing_utils
       mp_pose = mp.solutions.pose
       
       def calculate_angle(a,b,c):
              a = np.array(a) # First
              b = np.array(b) # Mid
              c = np.array(c) # End
        
              radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
              angle = np.abs(radians*180.0/np.pi)
        
              if angle >180.0:
                     angle = 360-angle
            
              return angle
       
       # Curl counter variables
       counter = 0 
       stage = None

       ## Setup mediapipe instance
       with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
              while cap.isOpened():
                     ret, frame = cap.read()
                     
                     # Recolor image to RGB
                     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     image.flags.writeable = False
              
                     # Make detection
                     results = pose.process(image)
              
                     # Recolor back to BGR
                     image.flags.writeable = True
                     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                     
                     # Extract landmarks
                     try:
                            landmarks = results.pose_landmarks.landmark
                     
                            # Get coordinates
                            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                            
                            # Calculate angle
                            angle = calculate_angle(shoulder, elbow, wrist)
                            
                            # Visualize angle
                            cv2.putText(image, str(angle), 
                                          tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                 )
                            Hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                            
                            
                            # Calculate angle
                            angle2 = calculate_angle(Hip, shoulder, elbow)
                            
                            # Visualize angle
                            cv2.putText(image, str(angle2), 
                                          tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                 )
                            
                            # tricep logic
                            
                            if angle < 120:
                                   stage = "down"
                            
                     
                            if angle > 120 and stage =='down':
                                   stage="up"
                                   counter +=1
                                   print(counter)
                     
                                   
                     except:
                            pass
                     
                     # Render curl counter
                     # Setup status box
                     cv2.rectangle(image, (0,0), (170,90), (0,0,0), -1)
                     
                     # Rep data
                     cv2.putText(image, 'REPS', (1,40), 
                            cv2.FONT_HERSHEY_SIMPLEX,0.6, (255,255,255), 1, cv2.LINE_AA)
                     cv2.putText(image, str(counter), 
                            (5,85), 
                            cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2, cv2.LINE_AA)
                     
                     # Stage data
                     cv2.putText(image, 'STAGE', (90,40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                     cv2.putText(image, stage, 
                            (78,85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                     
                     if angle2<168:
                            cv2.rectangle(image,(370,0),(639,60),(0,0,0),1)
                            cv2.putText(image,'Advice:shdr-180 ',(375,50),
                                   cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA
                                   )
                     elif angle<90:
                            cv2.rectangle(image,(370,0),(639,120),(0,0,0),1)
                            cv2.putText(image,'Advice:start-120 ',(375,50),
                                   cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA
                                   )
                     
                     # Render detections
                     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                          )               
                     
                     cv2.imshow('Mediapipe Feed', image)

                     if cv2.waitKey(10) & 0xFF == ord('q'):
                            break

              cap.release()
              cv2.destroyAllWindows()

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/login',methods=['POST','GET'])
def login():
       if request.method=='POST':
              res=""
              username=request.form['username']
              password=request.form['password']
       if username=="root" and password=="root":
              res="prepage"
       else:
              res="login"
       return redirect(url_for(res))
        

@app.route("/prepage")
def prepage():
       return render_template("flex.html")

@app.route('/biceps')
def biceps():
       return Response(gen_biceps(),mimetype="multipart/x-mixed-replace;boundary=frame")

@app.route('/shoulderpress')
def shoulderpress():
       return Response(gen_shoulder(),mimetype="multipart/x-mixed-replace;boundary=frame")

@app.route('/triceps')
def triceps():
       return Response(gen_triceps(),mimetype="multipart/x-mixed-replace;boundary=frame")

if __name__=="__main__":
    app.run(debug=True)
