#constants such as the name of the game window, FPS
GameName = "FlappyBird"
GameDirectory = "D:/FlappyBird-QLearning/FlapPyBird"
LaunchCommand = "D:/Dizertatie/FlappyBird-QLearning/startGame.bat"
GameScoreNumberAssets = "FlapPyBird/assets/sprites/"
GameOver = "gameover.png"
FPS = 30
WIDTH = 288
HEIGHT = 512
RENDER = True
#posible inputs according to http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
#for flappy birds the only posible input is W for jump
W = 0x11
inputsList = [W]

#Neural network parameters
inputLayer = 32292
hiddenLyaer = 5
outputLayer = 2
learningRate = 0.1
epochs = 0

#replay memory parameters
memorySize=100;
maxMemory = 10;
discount = 0.9

#logging variables
useFileLogging = True;
useConsoleLogging = True;



