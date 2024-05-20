# To Capture Frame
import cv2

# To process image array
import numpy as np

# import the tensorflow modules and load the model
import tensorflow as tf

model = tf.keras.models.load_model("converted_keras(1)/keras_model.h5")

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:
    # Reading / Requesting a Frame from the Camera 
    status, frame = camera.read()

    # if we were successfully able to read the frame
    if status:
        # Flip the frame
        frame = cv2.flip(frame, 1)

        # Resize the frame
        image = cv2.resize(frame, (224, 224))
        
        # Expand the dimensions
        image = np.array(image, dtype=np.float32)
        image = np.expand_dims(image, axis=0)
        
        # Normalize it before feeding to the model
        test1Image = image / 255
        
        # Get predictions from the model
        prediction = model.predict(test1Image)
        print(prediction)

        # Displaying the frames captured
        cv2.imshow('feed', frame)

        # Waiting for 1ms
        key = cv2.waitKey(1)
        
        # If space key is pressed, break the loop
        if key == 32:
            break

# Release the camera from the application software
camera.release()

# Close the open window
cv2.destroyAllWindows()
