import numpy as np
from PIL import ImageGrab
import cv2
import time
import imutils
import skimage.transform
import sys
import pytesseract
import Constants

templates = [None]*10
gameOverTemplate = cv2.imread(Constants.GameOver,0)
reward = 0
previousDigit =-1
def loadTemplates():
    for i in range(0,len(templates)):
        asset =Constants.GameScoreNumberAssets + str(i)+".png"
        templates[i]=cv2.imread(asset,0)

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
    for i in range(0,len(templates)):
        template = templates[i]
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.98
       
        loc = np.where( res >= threshold)

        for pt in zip(*loc[::-1]):
            gameOver = False
            if i == 1 and reward == 0:
                reward = 1
                previousDigit = i
            else:
                if reward !=0 and previousDigit !=i:
                    reward += 1
                    previousDigit = i
    ####### Testing purpose to see if it takes the image correctly
    if gameOver ==True:
        template = gameOverTemplate
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
       
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            #now it should be game over
            print("game over ")
            reward = 0
            previousDigit = -1
    print("reward is " + str(reward))
    winname = "OpenCV Render"
    cv2.namedWindow(winname)   
    cv2.moveWindow(winname, Constants.WIDTH,0)

    cv2.imshow(winname,templates[0])
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


