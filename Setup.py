import Controller
import Preprocessing.Preprocessing
import win32gui
import win32con
import Constants
import Utils.Logger
if __name__ == '__main__':
    hwndMain = Controller.Initialize()
    Preprocessing.Preprocessing.loadTemplates()
    previousScore = 0
    Utils.Logger.Logger.Log("Start of session")
    while 1==1:
        image = Preprocessing.Preprocessing.screen_record();
        score = Preprocessing.Preprocessing.getScore();
        if score > 0:
            Utils.Logger.Logger.Log(score)
            previousScore = score
        if score == 0 and previousScore != 0: 
            Utils.Logger.Logger.Log("End of episode")
            previousScore = score
        #Controller.PressKey(0x11)
        #Controller.ReleaseKey(0x11)

