import Controller
import Preprocessing.Preprocessing
import win32gui
import win32con
import Constants
import Utils.Logger
import QLearning.NeuralNetwork
import QLearning.ReplayMemory
import numpy as np
if __name__ == '__main__':
    hwndMain = Controller.Initialize()
    Preprocessing.Preprocessing.loadTemplates()
    replayMemory = QLearning.ReplayMemory.ReplayMemory()
    network =  QLearning.NeuralNetwork.Brain(1)
   
    firstLaunch = True
    previousState = None
    Utils.Logger.Logger.Log("Start of session")
    framesCount = 0
    while 1==1:
        image = Preprocessing.Preprocessing.ScreenRecord2048();
        score = Preprocessing.Preprocessing.getScore();
        if score != -1:
            #if the game did not end we skip some frames for effiency
            framesCount = framesCount + 1
            if(framesCount == 4):
                framesCount = 0
                #continue
        if(previousState != None):
            gameEnded = False
            if score == -1:
                gameEnded = True
            previousState.endEpisode = gameEnded
            # we need the next frame in order to be sure if the previous action resulted in ended game
            replayMemory.StoreInMemory(previousState)
            previousState = None
        if score == -1:
            Utils.Logger.Logger.Log("Start of episode")
            Controller.PressKey(0x11)
            Controller.ReleaseKey(0x11)
            firstLaunch = False
            if replayMemory.IsFull():
                replayMemory.PrintReplays()
                replayMemory.UpdateQValues();
                network.optimize(replayMemory)
                replayMemory.ClearMemory()
            continue
        continue
        state = np.dstack([image])
        qvalues = network.get_q_values(states=[state])
        qvalues = qvalues.flatten()
        action = np.argmax(qvalues)
        previousState = QLearning.ReplayMemory.State(state, qvalues, action, score,False)
        if score > 0:
            Utils.Logger.Logger.Log(score)
        if action == 0:
            Controller.PressKey(0x11)
            Controller.ReleaseKey(0x11)

