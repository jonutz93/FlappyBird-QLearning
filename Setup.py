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
    replayMemory = QLearning.ReplayMemory.ReplayMemory()
    network =  QLearning.NeuralNetwork.Brain(1)
    hwndMain = Controller.Initialize()
    Preprocessing.Preprocessing.loadTemplates()
    previousScore = -1
    Utils.Logger.Logger.Log("Start of session")
    while 1==1:
        image = Preprocessing.Preprocessing.screen_record();
        score = Preprocessing.Preprocessing.getScore();
        if score == -1:
            Utils.Logger.Logger.Log("Start of episode")
            Controller.PressKey(0x11)
            Controller.ReleaseKey(0x11)
            endOfEpisode = True;
            if replayMemory.IsFull():
                replayMemory.PrintReplays()
                #replayMemory.updateQValues();
                #network.optimize(replayMemory)
                replayMemory.clearMemory()
            continue
        endOfEpisode = False
        state = np.dstack([image])
        qvalues = network.get_q_values(states=[state])
        qvalues = qvalues.flatten()
        action = np.argmax(qvalues)
        replayMemory.StoreInMemory(state,qvalues,action,score,endOfEpisode)
        if score > 0:
            Utils.Logger.Logger.Log(score)
            previousScore = score
        if action == 0:
            Controller.PressKey(0x11)
            Controller.ReleaseKey(0x11)

