import numpy as np
from PIL import ImageGrab
import cv2
import time
import imutils
import skimage.transform
import sys
import time

import Constants

templates = [None]*10
gameOverTemplate = cv2.imread(Constants.GameOver,0)
reward = 0
episode = 0
def loadTemplates():
    for i in range(0,len(templates)):
        asset =Constants.GameScoreNumberAssets + str(i)+".png"
        templates[i]=cv2.imread(asset,0)
#function that gets the score from an image
def getScreenScore(image):
    lower_white= np.array([0,0,0])
    upper_white = np.array([100,100,100])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            if mask[x, y] ==0:
                mask[x, y] = 255
            else:
                if mask[x, y] ==255:
                    mask[x, y] =0
    numbers = {}
    for i in range(0,len(templates)):
        template = templates[i]
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(mask,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.97
       
        loc = np.where( res >= threshold)

        for pt in zip(*loc[::-1]):
            numbers[pt[0]] = i
    numbers = sorted(numbers.items(), key=lambda x: x[0])
    scoreNumber = -1
    if len(numbers) > 0:
        scoreNumber = 0
        for key, value in numbers:
            scoreNumber = (scoreNumber*10) +value
    return scoreNumber
def updateReward(newReward):
    global reward,previousReward
    if newReward > reward :
        reward = newReward
    if newReward == 0 and reward !=0:
        reward = 0
    if newReward == -1:
        reward = -1
def getScore():
    global reward
    return reward;
def screen_record(): 
    global reward
    last_time = time.time()
    # WIDTH*HEIGHT windowed mode
    printscreen =  np.array(ImageGrab.grab(bbox=(0,25,Constants.WIDTH-12.5,Constants.HEIGHT-20)))
    miniImage = printscreen[50:150,0:Constants.WIDTH];
    updateReward(getScreenScore(miniImage));
    #change to gray
    printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
    #half the image size
    printscreen = cv2.resize(printscreen, (0,0), fx=0.5, fy=0.5) 
    if Constants.RENDER == True:
        winname = "OpenCV Render"
        cv2.namedWindow(winname)   
        cv2.moveWindow(winname, Constants.WIDTH,0)

        cv2.imshow(winname,printscreen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
    return printscreen


