import Controller
import Preprocessing.Preprocessing
import win32gui
import win32con
import Constants
if __name__ == '__main__':
    hwndMain = Controller.Initialize()
    Preprocessing.Preprocessing.loadTemplates()
    while 1==1:
        Preprocessing.Preprocessing.screen_record()
        #Controller.PressKey(0x11)
        #Controller.ReleaseKey(0x11)

