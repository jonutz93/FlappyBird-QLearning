import Controller
import Preprocessing
import win32gui
import win32con
import Constants
def RaiseWindowNamed(nameRe):
  import win32gui
  # start by getting a list of all the windows:
  cb = lambda x,y: y.append(x)
  wins = []
  win32gui.EnumWindows(cb,wins)

  # now check to see if any match our regexp:
  tgtWin = -1
  for win in wins:
    txt = win32gui.GetWindowText(win)
    if nameRe == txt:
      tgtWin=win
      break

  if tgtWin>=0:
    win32gui.SetForegroundWindow(tgtWin)
if __name__ == '__main__':
    #RaiseWindowNamed(Constants.GameName)

    while 1==1:
        Preprocessing.screen_record()
        Controller.PressKey(0x11)
        Controller.ReleaseKey(0x11)

