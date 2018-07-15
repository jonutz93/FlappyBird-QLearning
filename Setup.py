import Controller
import Preprocessing.Preprocessing
import win32gui
import win32con
import Constants
if __name__ == '__main__':
    hwndMain = Controller.Initialize()
    Preprocessing.Preprocessing.loadTemplates()
    previousScore = 0
    while 1==1:
        image = Preprocessing.Preprocessing.screen_record();
        score = Preprocessing.Preprocessing.getScore();
        if score > 0:
            print(score)
            previousScore = score
        if score == 0 and previousScore!=0: 
            print("end of episode")
            previousScore = score
        #Controller.PressKey(0x11)
        #Controller.ReleaseKey(0x11)

