#this works only on win32
import ctypes
import time
import win32con
from win32gui import GetWindowText, GetForegroundWindow
import win32gui
import os
import subprocess
import asyncio
#internal imports
import Constants
import Utils.Logger
import sys
SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

#function that presses a key on the active window.
#Future me should find to send the input to a specific window istead of the active one
#it only sends the input if the active window is Constants.GameName
def PressKey(hexKeyCode):
    if GetWindowText(GetForegroundWindow()) == Constants.GameName:
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
        x = Input( ctypes.c_ulong(1), ii_ )
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        time.sleep(1/Constants.FPS)

def ReleaseKey(hexKeyCode):
    if GetWindowText(GetForegroundWindow()) == Constants.GameName:
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
        x = Input( ctypes.c_ulong(1), ii_ )
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
        time.sleep(1/Constants.FPS)

#this is work in progress.
#It should somehow disable the movement of the game window
#
oldWndProc = {}
def MyWndProc(self, hWnd, msg, wParam, lParam):
         # Display what we've got.
         Utils.Logger.Logger.Log(self.msgdict.get(msg), msg, wParam, lParam)
         
         # Restore the old WndProc.  Notice the use of wxin32api
         # instead of win32gui here.  This is to avoid an error due to
         # not passing a callable object.
         if msg == win32con.WM_DESTROY: 
             win32api.SetWindowLong(self.GetHandle(), 
                                    win32con.GWL_WNDPROC, 
                                    self.oldWndProc) 
         if msg == win32con.WM_SYSCOMMAND:
             if wParam ==  win32con.SC_MOVE:
                 return
         # Pass all messages (in this case, yours may be different) on
         # to the original WndProc
         return win32gui.CallWindowProc(self.oldWndProc,
                                        hWnd, msg, wParam, lParam)

def RaiseWindowNamed(nameRe):
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
def Initialize():
    # open the game process
    #the window should be set on the top left
    proc = subprocess.Popen(Constants.LaunchCommand, shell=True,
             stdin=None, stdout=None, stderr=None, close_fds=True)
    hwndMain = win32gui.FindWindow(None, Constants.GameName)
    while hwndMain == 0:
        #The game is not running
        #wait for it to start running
        hwndMain = win32gui.FindWindow(None, Constants.GameName)
        time.sleep(1/Constants.FPS)
    win32gui.SetWindowPos(hwndMain, win32con.HWND_TOPMOST, 0, 0, 0, 0, False)
    win32gui.MoveWindow(hwndMain, -5, -5, Constants.WIDTH, Constants.HEIGHT, True)
    RaiseWindowNamed(Constants.GameName)
    #win32gui.SetForegroundWindow(hwndMain)
    #Set the WndProc to our function
    oldWndProc = win32gui.SetWindowLong(hwndMain,
                                        win32con.GWL_WNDPROC,
                                        MyWndProc)
    return hwndMain