import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
import streamlit as st
from PIL import Image, ImageOps
import streamlit as st

# Streamlit Started =====>
st.set_page_config(layout='wide')
# st.image('maths.jpg')

# Load and display the image with a border
image = Image.open('maths.jpg')
border_width = 10  # Adjust the border width as needed
image_with_border = ImageOps.expand(image, border=border_width, fill='black')
st.image(image_with_border)

col1, col2 = st.columns([3,2])

with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    output_text = st.title('Answer')
    answer = st.subheader('')

# Streamlit Ended <=====
model = genai.GenerativeModel('gemini-1.5-flash')
genai.configure(api_key="AIzaSyD-edDjXingTU2SjjgrwPHnEEPVO96_e5U")




# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(3,1100) # 3 = width 
cap.set(4,600) # 4 = height

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def GetHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmlist = hand1["lmList"]  # List of 21 landmarks for the first hand

        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand1)
        # print(f'H1 = {fingers.count(1)}', end=" ")  # Print the count of fingers that are up
        print(fingers)

        print(" ")  # New line for better readability of the printed output

        return fingers,lmlist
    else:
        return None


def Draw(info, prev_pos,canvas):
    fingers, lmlist = info
    current_pos = None
    
    if fingers == [0,1,0,0,0] or fingers == [1,1,0,0,0]:

        current_pos = lmlist[8][0:2]
        if prev_pos == None: prev_pos = current_pos

        cv2.line(canvas, current_pos, prev_pos , (0,255,255), 10)

    elif fingers == [1,1,1,1,1]:
        canvas = np.zeros_like(img)

    return current_pos, canvas

def SendToGemini(model, canvas):

    #converting numpy canvas to Image b/c gemini process only images
    image = Image.fromarray(canvas)
    response = model.generate_content(["Solve this Maths problem", image])
    print(response.text)
    return response.text 


prev_pos = None
canvas = None
image_combine = None
response = None

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    img = cv2.flip(img , 1) #1 = horizontal y-axis

    if canvas is None: 
        canvas = np.zeros_like(img)
        # image_combine = img.copy()

    info = GetHandInfo(img)

    if info:
        fingers, lmlist = info
        print(fingers)
        prev_pos, canvas = Draw(info, prev_pos, canvas)
        
        if fingers == [1,1,1,1,0]:
           response = SendToGemini(model,canvas)
           answer.subheader(response)

    #combining image and canvas
    image_combine = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combine, channels = 'BGR')
    


    # Display the image in a window
    # cv2.imshow("Image", img)
    # cv2.imshow("canvas", canvas)
    # cv2.imshow("image_combine", image_combine)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)