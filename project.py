import random
import deepface
import cv2
from deepface import DeepFace


emoji_dic ={
    "happy":['😁','😂','🤣','😊','😀','😄'],
    "sad":['😒','😫','🙁','😖','😞','☹️','😟','😩'] , 
    "neutral":['😐'],
    "angry":['😤','😡','👺'],
    'na' :['na']                           
}

cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Camera is not opened")
    exit()

frame_count = 0
emotion = "na"
string_a = ""
# Infinite loop to capture and display video frames
while cap.isOpened():
    # Capture the next frame
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Frame capture failed")
        break
    img = frame
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    
    if len(face) ==1 :
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        
        if frame_count % 10 == 0 :   
            try :
                result = DeepFace.analyze(frame[y:y+h , x:x+h],
                                    actions = ['emotion'])
                dictionary = result[0]['emotion']
                emotion = ( max( dictionary , key = dictionary.get ))

                


            except :
                pass
        
        if emotion =='fear' :
            emotion  = 'sad'
        # Define the text, position, font, font scale, color, and thickness
        text = str( emotion )
        position = (x+w , y+h)  # (x, y) coordinates of the top-left corner of the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (255, 255, 255)  # Text color in BGR format (white in this example)
        thickness = 2  # Text thickness
        # font_face = 'Segoe UI Emoji'

        # Draw the text on the image
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        string_a += random.choice( emoji_dic[emotion])

    with open('emoji.txt' , 'w' ,  encoding='utf-8') as file :
        file.write(string_a )
    # Display the frame
    cv2.imshow("Frame", frame)

    # Press 'q' to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    frame_count += 1

# Release the camera
cap.release()

# Close all windows
cv2.destroyAllWindows()
