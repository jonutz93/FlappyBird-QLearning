# Simple Snake Game in Python 3 for Beginners
# By @TokyoEdTech

import time
import random
import numpy as np

import QLearning.NeuralNetwork
import QLearning.ReplayMemory
import Utils.Logger
import Controller
import Constants
"""
 Pygame base template for opening a window
 
 Sample Python/Pygame Programs
 Simpson College Computer Science
 http://programarcadegames.com/
 http://simpson.edu/computer-science/
 
 Explanation video: http://youtu.be/vRB_983kUMc
"""
 
import pygame
 
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
 
pygame.init()
 

blockSize = 20
w, h = 10, 10;

# Set the width and height of the screen [width, height]
size = (w*blockSize, h*blockSize)
screen = pygame.display.set_mode(size)
 
pygame.display.set_caption("Snake")
 
# Loop until the user clicks the close button.
done = False
 
# Used to manage how fast the screen updates
clock = pygame.time.Clock()



snakeImg = pygame.image.load('snake.png')
snakeImg = pygame.transform.scale(snakeImg, (blockSize, blockSize))
cherryImg = pygame.image.load('cherry.png')
cherryImg = pygame.transform.scale(cherryImg, (blockSize, blockSize))


class Game(object):
    def __init__(self):
        self.reset()
    def reset(self):
        global w,h
        self.snakeList = []
        self.snakeLength = 1
        self.game_over = False
        self.score = 0
        self.headX = int(w/2)
        self.headY = int(h/2)      
        self.goDown =  self.goUp =  self.goLeft =  self.goRight = False
        self.loadLevel()
        self.generateFood()
    def generateFood(self):
        n = 1 #random.randint(1,w*h)
        count = 0
        while True:
            for i in range(len(self.Matrix)):
                for j in range(len(self.Matrix[i])):
                    if self.Matrix[i][j] == 0:
                        count = count + 1
                        if count == n:
                            self.food = (i,j)
                            return
    def loadLevel(self):
        # box
        self.Matrix = [[0 for x in range(w)] for y in range(h)]
        for i in range(len(self.Matrix)):
            self.Matrix[i][0] = -1
            self.Matrix[i][len(self.Matrix)-1] = -1
            self.Matrix[0][i] = -1
            self.Matrix[len(self.Matrix)-1][i] = -1

    def getInput(self):
        for event in pygame.event.get():
           if event.type == pygame.QUIT:
                game_over = True
           if event.type == pygame.KEYDOWN:
               self.setInput(event.key == pygame.K_UP, event.key == pygame.K_LEFT, event.key == pygame.K_DOWN, event.key == pygame.K_RIGHT)
    def setInput(self, W,A,S,D):
        if A and self.goRight == False:
            self.goDown = self.goUp = self.goLeft = self.goRight = False
            self.goLeft = True
        elif D and self.goLeft == False:
            self.goDown = self.goUp = self.goLeft = self.goRight = False
            self.goRight = True
        elif W and self.goDown == False:
            self.goDown = self.goUp = self.goLeft = self.goRight = False
            self.goUp = True
        elif S and self.goUp == False:
            self.goDown = self.goUp = self.goLeft = self.goRight = False
            self.goDown = True
    def update(self):
        global w,h
        if self.goDown:
           self.headY = self.headY+1;
           if(self.headY >= w):
              self.headY = 0
        if self.goLeft:
           self.headX = self.headX-1;
           if(self.headX < 0):
              self.headX = h-1
        if self.goUp:
           self.headY = self.headY-1;
           if(self.headY < 0):
              self.headY = w-1
        if self.goRight:
           self.headX = self.headX+1;
           if(self.headX >= h):
              self.headX = 0

        if self.headX == self.food[0] and self.headY == self.food[1]:
            #self.snakeLength = self.snakeLength +1
            self.generateFood()
            self.score = self.score + 1
        elif self.Matrix[self.headX][self.headY]  == -1:
            self.game_over = True
            return
        #reset the matrix
        self.loadLevel()

        self.snakeList.append((self.headX,self.headY))
        if len(self.snakeList) > self.snakeLength:
            del self.snakeList[0]

        self.Matrix[self.headX][self.headY] = 1 #head is 1
        self.Matrix[self.food[0]][self.food[1]] = 2 # food
        for x in self.snakeList[:-1]:
            if x[0] == self.headX and x[1] == self.headY:
                self.game_over = True
            self.Matrix[x[0]][x[1]] = 3 # components
    def render(self):
        global blockSize,screen,score_font
        yellow = (255, 125, 102)
        value = score_font.render("Score: " + str(self.score), True, yellow)
        screen.blit(value, [0, 0])
        for i in range(len(game.Matrix)):
            for j in range(len(game.Matrix[i])):
               if self.Matrix[i][j] == 1 or self.Matrix[i][j] == 3 :
                    screen.blit(snakeImg, (blockSize*i,blockSize*j))
               if self.Matrix[i][j] == 2:
                    screen.blit(cherryImg, (blockSize*i,blockSize*j))
               if self.Matrix[i][j] == -1:
                    pygame.draw.rect(screen,RED,(blockSize*i,blockSize*j,blockSize,blockSize))

bestScore = 0
game = Game()
runAI = True
replayMemory = QLearning.ReplayMemory.ReplayMemory()
network =  QLearning.NeuralNetwork.Brain(1,w, h)

epsilon_greedy = QLearning.NeuralNetwork.EpsilonGreedy(start_value=1.0,
                                            end_value=0.1,
                                            num_iterations=30000,
                                            num_actions=4,
                                            epsilon_testing=0.01)

# The learning-rate for the optimizer decreases linearly.
learning_rate_control = QLearning.NeuralNetwork.LinearControlSignal(start_value=1e-3,
                                                             end_value=1e-5,
                                                             num_iterations=5e6)

            # The loss-limit is used to abort the optimization whenever the
            # mean batch-loss falls below this limit.
loss_limit_control = QLearning.NeuralNetwork.LinearControlSignal(start_value=0.1,
                                                          end_value=0.015,
                                                          num_iterations=5e6)

            # The maximum number of epochs to perform during optimization.
            # This is increased from 5 to 10 epochs, because it was found for
            # the Breakout-game that too many epochs could be harmful early
            # in the training, as it might cause over-fitting.
            # Later in the training we would occasionally get rare events
            # and would therefore have to optimize for more iterations
            # because the learning-rate had been decreased.
max_epochs_control = QLearning.NeuralNetwork.LinearControlSignal(start_value=5.0,
                                                          end_value=10.0,
                                                          num_iterations=5e6)

score_font = pygame.font.SysFont("comicsansms", 35)
framesCount = network.get_count_states()
render = True
#count_states = network.model.get_count_states()
# -------- Main Program Loop -----------
while not done:
    # --- Main event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    Utils.Logger.Logger.Log("Start of episode")
    game.reset()
    previsousScore = 0
    reward = 0
    logCounter =0
    while not game.game_over:
        screen.fill(BLACK)
        epsilon = 0
        if runAI ==True:
            state = np.dstack([game.Matrix])
            qvalues = network.get_q_values(states=[state])
            qvalues = qvalues.flatten()
            action = np.argmax(qvalues)
            action, epsilon = epsilon_greedy.get_action(q_values=qvalues,
                                                                 iteration = framesCount,
                                                                 training=True)
            framesCount = framesCount+1

            game.setInput(action == 3,action == 0,action == 2, action == 1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
        else:
            game.getInput()

        game.update()

        if runAI:
            if previsousScore < game.score: 
                reward = 10
                previsousScore = game.score
            elif game.game_over == True:
                reward = -10
            else:
                reward = -1
            #reward = game.score
            if framesCount%1 ==0:
                logCounter = logCounter +1
                Utils.Logger.Logger.Log("Frame:" + str(logCounter) + "q" + str(qvalues) + " action:" + str(action) + " score:" + str(game.score) +"(" + str(bestScore) +") epsilon:" + str(epsilon))
            previousState = QLearning.ReplayMemory.State(state, qvalues, action,  reward, game.game_over)
            replayMemory.StoreInMemory(previousState)
            count_episodes = network.increase_count_states()
       # --- Drawing code should go here
        if render:
            game.render()
            yellow = (255, 125, 102)
            value = score_font.render("Best: " + str(bestScore), True, yellow)
            screen.blit(value, [0, 50])
            pygame.display.flip()
       # --- Limit to 60 frames per second
        clock.tick(600)
    if(game.score >bestScore):
        bestScore = game.score
    #if replayMemory.IsFull():
    if True:
        replayMemory.PrintReplays()
        replayMemory.UpdateQValues();
        network.optimize(replayMemory)
        logCounter = 0
        for x in replayMemory.memory:
            logCounter = logCounter +1
            Utils.Logger.Logger.Log("Frame:" + str(logCounter)+ str(x.qValues));
        replayMemory.ClearMemory()
   
# Close the window and quit.
pygame.quit()