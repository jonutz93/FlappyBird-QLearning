import Constants
import Utils
class State(object):
    def __init__(self, state, qValues, action, reward, endEpisode):
         self.state = state
         self.action = action
         self.qValues = action
         self.reward = reward
         self.endEpisode = endEpisode

class ReplayMemory(object):
    def __init__(self):
        self.maxMemory = Constants.maxMemory
        self.memory = list()
        self.discount = Constants.discount
    def StoreInMemory(self, state, qValues, action, reward, end_episode):
        # Save a state to memory
        newState = State(state, qValues, action, reward, end_episode)
        self.memory.append(newState)
    def PrintReplays(self):
        for current in range(len(self.memory)):
            currentState = self.memory[current]
            Utils.Logger.Logger.Log("Index" + str(current) +" action: " + str(currentState.action) + " reward: " + str(currentState.reward) + " end of episode:" + str(currentState.endEpisode))
    def ClearMemory(self):
        self.memory.clear()
    def IsFull(self):
        return len(self.memory) > self.maxMemory
    def get_batch(self, model, batch_size=10):

        # How many experiences do we have?
        len_memory = len(self.memory)

        # Calculate the number of actions that can possibly be taken in the game.
        num_actions = model.output_shape[-1]

        # Dimensions of our observed states, ie, the input to our model.
        env_dim = self.memory[0].state.shape[1]

        # We want to return an input and target vector with inputs from an observed state.
        inputs = np.zeros((min(len_memory, batch_size), env_dim))

        # ...and the target r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken but also
        # for the other possible actions. The actions not take the same value as the prediction to not affect them
        targets = np.zeros((inputs.shape[0], num_actions))

        # We draw states to learn from randomly
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            """
            Here we load one transition <s, a, r, s’> from memory
            state_t: initial state s
            action_t: action taken a
            reward_t: reward earned r
            state_tp1: the state that followed s’
            """
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

            # We also need to know whether the game ended at this state
            game_over = self.memory[idx][1]

            # add the state s to the input
            inputs[i:i + 1] = state_t

            # First we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0)
            targets[i] = model.predict(state_t)[0]

            """
            If the game ended, the expected reward Q(s,a) should be the final reward r.
            Otherwise the target value is r + gamma * max Q(s’,a’)
            """
            #  Here Q_sa is max_a'Q(s', a')
            Q_sa = np.max(model.predict(state_tp1)[0])

            # if the game ended, the reward is the final reward
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # r + gamma * max Q(s’,a’)
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets

