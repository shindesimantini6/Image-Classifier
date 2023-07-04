# Image-Classifier

Trained a model to classify images using a Convolutional Neural Network. The multiple images of multiple objects were taken at different angles and then trained. The model shows very good accuracy for certain objects. If bottles were taken of several types, the bottles were stored and trained for each of its type. This was to train the model for each of the special characteristics of the bottles. 

Types of images (e.g.):
1. apple
2. biketool
3. bodycream
4. bottle_1
5. glasses
6. plant
7. scarf

Results:
Some bottles were classified as other types of bottles, but they had some similarity to it. 

Use:
- Run `python3 imageclassifier/src/capture.py models.py` to start the camera and click `space` everytime you want the model to return a prediction. 
- exit the program with `q`

NOTE: A video will be updated soon. 

Requirements:
- Python 3.8
- opencv-python
