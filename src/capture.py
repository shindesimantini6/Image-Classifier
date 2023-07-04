import sys
import logging
import os
import cv2
from utils import write_image, key_action, init_cam, predict_frame
import time


if __name__ == "__main__":

    # folder to write images to
    out_folder = sys.argv[1]

    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    logging.getLogger().setLevel(logging.INFO)
   
    # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None

    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            # draw a [224x224] rectangle into the frame, leave some space for the black border 
            offset = 2
            width = 224
            x = 160
            y = 120
            cv2.rectangle(img=frame, 
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 0, 0), 
                          thickness=2
            )     
            
            # get key event
            key = key_action()

            if key == 'space':
                # write the image without overlay
                # extract the [224x224] rectangle out of it
                image = frame[y:y+width, x:x+width, :]
                # write_image(out_folder, image)
                prediction, number = predict_frame(image, '/home/shinde/Documents/trainings/Spiced_Academy/Github/tahini-tensor-student-code/spiced_projects/week9/models/trained_for_all_objects.h5')
                # total_value = prediction + ":" + '\n' + str(number)
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # org
                org = (00, 170)

                org1 = (00, 185)
                
                # fontScale
                fontScale = 0.5
                
                # Red color in BGR
                color = (0, 0, 255)
                
                # Line thickness of 2 px
                thickness = 1

                # Using cv2.putText() method
                cv2.putText(image, prediction, org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
                
                # Using cv2.putText() method
                cv2.putText(image, str(number), org1, font, 
                fontScale, color, thickness, cv2.LINE_AA)

                cv2.imshow('Prediction: ', image)


            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)
            
    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        # cv2.destroyAllWindows()
