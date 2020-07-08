#constants such as the name of the game window, FPS
GameName = "2048"
GameDirectory = "D:/FlappyBird-QLearning/FlapPyBird"
LaunchCommand = "D:/Dizertatie/FlappyBird-QLearning/startGame.bat"
GameScoreNumberAssets = "FlapPyBird/assets/sprites/"
GameOver = "gameover.png"
FPS = 30
WIDTH = 900
HEIGHT = 700
RENDER = True
#posible inputs according to http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
#for flappy birds the only posible input is W for jump
key_W = 0x11
key_A = 0x1E
key_D = 0x20
key_S = 0x1F

#Neural network parameters
inputLayer = 160
hiddenLyaer = 50
outputLayer = 4
learningRate = 0.1
epochs = 20
''' flappy birds
#image size
state_height = 234
# Width of each image-frame in the state.
state_width = 138
'''

'''  2048
state_height = 4
state_width = 4
'''
#sdsadsa
'''
'''

state_height = 4
state_width = 4

#replay memory parameters
maxMemory = 1000;
discount = 0.9

#logging variables
useFileLogging = False;
useConsoleLogging = True;



