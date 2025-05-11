import cv2 #for image/video processing
import numpy as np #for working with arrays and color range definitions
import matplotlib.pyplot as plt # To display images without using cv2.imshow()

#Opens default webcam (device 0)
cap = cv2.VideoCapture(0)

#Enables interactive mode in matplotlib for real-time updates.
plt.ion()
#Creates a figure and axis for plotting frames
fig,ax=plt.subplots()

stop_loop = False

def on_key(event):  #is a function that will be called whenever a key is pressed.
    global stop_loop
    if event.key == 'q':    #checks if the key pressed is 'q'.
        stop_loop = True    # sets a flag to True, which can stop the while loop in your main program.

#fig.canvas.mpl_connect() listens for events on the Matplotlib canvas (i.e., the figure window)
# and triggers the on_key function whenever a key is pressed.
fig.canvas.mpl_connect('key_press_event',on_key)

# Continuously reads frames from webcam.
# ret is True if a frame is successfully captured.
# frame is the actual image from the webcam.
# If no frame is read, break the loop.
while not stop_loop:
    ret, frame = cap.read()
    if not ret:
        break

    #Converts the frame from BGR to HSV color space
    #HSV makes it easier to isolate specific colors
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #--------------------------------Colors upper and lower HSV Ranges----------------------------------------
    colors = {
        "Red": [
            (np.array([0,120,70]),np.array([10,225,225])),
            (np.array([170,120,70]),np.array([180,225,225]))
        ],
        "Blue": [
            (np.array([100,150,0]),np.array([140,255,255]))
        ],
        "Green": [
            (np.array([40, 70, 70]), np.array([80, 255, 255]))
        ],
        "Yellow": [
            (np.array([20, 100, 100]), np.array([30, 255, 255]))
        ]
    }

    for color_name, ranges in colors.items():
        mask = None

        for lower,upper in ranges:
            current_mask = cv2.inRange(hsv,lower,upper)
            mask = current_mask if mask is None else (mask | current_mask)

        #Find contours for the combined mask
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 200:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,f"{color_name} Object",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)




    #Converts result to RGB so matplotlib shows correct colors.
    result_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    #Clears the previous frame
    #Shows the current processed frame
    #Updates the plot in real time with a tiny pause 0.001 sec
    ax.clear()
    ax.imshow(result_rgb)
    ax.set_title("RealTime Color Detection : 'q' to quit")
    ax.axis('off')
    plt.pause(0.001)

#Releases the webcam
#Turns off interactive mode
#Keeps the final plot window Open
cap.release()
plt.ioff()
plt.show()
