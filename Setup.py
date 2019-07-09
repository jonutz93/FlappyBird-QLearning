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
    epsilon_greedy = QLearning.NeuralNetwork.EpsilonGreedy(start_value=1.0,
                                            end_value=0.1,
                                            num_iterations=1e6,
                                            num_actions=4,
                                            epsilon_testing=0.01)
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
            Controller.PressAndReleaseKey(Constants.key_A)
            if replayMemory.IsFull():
                replayMemory.PrintReplays()
                replayMemory.UpdateQValues();
                network.optimize(replayMemory)
                replayMemory.ClearMemory()
            Controller.CloseWindow(hwndMain)
            hwndMain = Controller.Initialize()
            continue
            
        state = np.dstack([image])
        qvalues = network.get_q_values(states=[state])
        qvalues = qvalues.flatten()
        action = np.argmax(qvalues)
        # Determine the action that the agent must take in the game-environment.
            # The epsilon is just used for printing further below.
        framesCount = framesCount+1
        action, epsilon = epsilon_greedy.get_action(q_values=qvalues,
                                                             iteration = framesCount,
                                                             training=True)
        previousState = QLearning.ReplayMemory.State(state, qvalues, action, score,False)
        if(action == 0):
            Controller.PressAndReleaseKey(Constants.key_A)
        elif(action == 1):
            Controller.PressAndReleaseKey(Constants.key_D)
        elif(action == 2):
            Controller.PressAndReleaseKey(Constants.key_S)
        elif(action == 3):
            Controller.PressAndReleaseKey(Constants.key_W)
        else :
            continue
        #if score > 0:
        #    Utils.Logger.Logger.Log(score)
        #if action == 0:
        #    Controller.PressKey(0x11)
        #    Controller.ReleaseKey(0x11)
         
