import cv2

#get image
img = cv2.imread('lena.PNG')

#define list for class names
classNames = []
classFile = 'coco.names'

#open coco names as f
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n') #split on every new line
print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#settings for cv2
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#doing the detection, output is detected name from coco and the coordinates of the box
classIds, confs, bbox = net.detect(img,confThreshold=0.5) #confThreshold is how sure about detection it should be
print(classIds, bbox)

for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox): #going through all class IDs, confidences and bbox
    cv2.rectangle(img,box,color=(0,255,0),thickness=3) #drawing the box (image, coordinates, color, thickness)
    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) #add text and take is form the coconames, set location, font, scale, color, thickness

#show image
cv2.imshow("Output", img)
cv2.waitKey(0)