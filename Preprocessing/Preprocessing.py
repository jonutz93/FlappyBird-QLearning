import numpy as np
from PIL import ImageGrab
import cv2
import time
import imutils
import skimage.transform
import sys
import pytesseract
import time

import Constants

templates = [None]*10
gameOverTemplate = cv2.imread(Constants.GameOver,0)
reward = 0
previousDigit =-1
episode = 0
initialTime = time.time()
currentTIme = time.time() 
def loadTemplates():
    for i in range(0,len(templates)):
        asset =Constants.GameScoreNumberAssets + str(i)+".png"
        templates[i]=cv2.imread(asset,0)

def resetGame():
    global episode,reward,previousDigit,initialTime
    print("game over")
    print("reward for episode: "+str(episode)+" is " + str(reward))
    reward = 0
    previousDigit = -1
    episode+=1
    initialTime = time.time()

def checkReward():
    global currentTIme,initialTime,reward
    currentTIme = time.time() 
    # very dificult to get the score from the image
    # so i'll make a workaround base on time
    # at second 4 it should have passed the first pipe
    # then aprox 1.2 seconds on each pipe
    difference = currentTIme - initialTime
    if(difference > 5 and reward == 0 ):
        initialTime = currentTIme
        updateReward(1)
    else:
        if (difference >1.2 and reward!=0):
            initialTime = currentTIme
            updateReward(reward+1)
def updateReward(newReward):
    global reward
    print("new reward is "+ str(newReward))
    reward = newReward

def screen_record(): 
    global reward,prevGray,previousDigit
    last_time = time.time()
    # WIDTH*HEIGHT windowed mode
    printscreen =  np.array(ImageGrab.grab(bbox=(0,25,Constants.WIDTH-12.5,Constants.HEIGHT-20)))
   
    img_gray = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)
    score = 0 
    #make Template Matching with the numbers from assets
    newGame = False
    #reward loop
    #Currently works well until 10. After 10 the reward will increase a lot faster.
    #TO DO: Fix this
    gameOver = True
    if previousDigit !=-1:
        checkReward()
    for i in range(0,len(templates)):
        template = templates[i]
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.98
       
        loc = np.where( res >= threshold)

        for pt in zip(*loc[::-1]):
            gameOver = False
            previousDigit = i
            if i == 1 and reward == 0:
                #updateReward(1)
                #do nothing for know
                x=0
            else:
                if reward !=0 and previousDigit !=i:
                    #updateReward(reward+1)
                    #do nothing for know
                    x=0
    ####### Testing purpose to see if it takes the image correctly
    if gameOver ==True:
        template = gameOverTemplate
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
       
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            #now it should be game over
            if previousDigit != -1:
                resetGame()
    winname = "OpenCV Render"
    cv2.namedWindow(winname)   
    cv2.moveWindow(winname, Constants.WIDTH,0)

    cv2.imshow(winname,templates[0])
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


