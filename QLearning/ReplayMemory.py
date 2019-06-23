import Constants
import Utils
import numpy as np
import random
class State(object):
    def __init__(self, state, qValues, action, reward, endEpisode):
         self.state = state
         self.action = action
         self.qValues = qValues
         self.reward = reward
         self.endEpisode = endEpisode

class ReplayMemory(object):
    def __init__(self):
        self.maxMemory = Constants.maxMemory
        self.memory = list()
        self.discountFactor = Constants.discount
    def StoreInMemory(self, state):
        # Save a state to memory
        self.memory.append(state)
    def UpdateQValues(self):
        """
        Update all Q-values in the replay-memory.
        
        When states and Q-values are added to the replay-memory, the
        Q-values have been estimated by the Neural Network. But we now
        have more data available that we can use to improve the estimated
        Q-values, because we now know which actions were taken and the
        observed rewards. We sweep backwards through the entire replay-memory
        to use the observed data to improve the estimated Q-values.
        """
        
        # Copy old Q-values so we can print their statistics later.
        # Note that the contents of the arrays are copied.
        #self.q_values_old[:] = self.mmemory q_values[:]
        
        # Process the replay-memory backwards and update the Q-values.
        # This loop could be implemented entirely in NumPy for higher speed,
        # but it is probably only a small fraction of the overall time usage,
        # and it is much easier to understand when implemented like this.
        for k in reversed(range(len(self.memory)-1)):
            # Get the data for the k'th state in the replay-memory.
            action = self.memory[k].action
            reward = self.memory[k].reward
            end_episode = self.memory[k].endEpisode
        
            # Calculate the Q-value for the action that was taken in this state.
            if end_episode:
                # If the agent lost a life or it was game over / end of episode,
                # then the value of taking the given action is just the reward
                # that was observed in this single step. This is because the
                # Q-value is defined as the discounted value of all future game
                # steps in a single life of the agent. When the life has ended,
                # there will be no future steps.
                actionValue = reward
            else:
                # Otherwise the value of taking the action is the reward that
                # we have observed plus the discounted value of future rewards
                # from continuing the game. We use the estimated Q-values for
                # the following state and take the maximum, because we will
                # generally take the action that has the highest Q-value.
                actionValue = reward + self.discountFactor * np.max(self.memory[k+1].qValues)
        
            # Error of the Q-value that was estimated using the Neural Network.
            # self.estimation_errors[k] = abs(action_value - self.q_values[k, action])
        
            # Update the Q-value with the better estimate.
            self.memory[k].qValues[action] = actionValue
    def PrintReplays(self):
        for current in range(len(self.memory)):
            currentState = self.memory[current]
            Utils.Logger.Logger.Log("Index" + str(current) +" action: " + str(currentState.action) + " reward: " + str(currentState.reward) + " end of episode:" + str(currentState.endEpisode))
    def ClearMemory(self):
        self.memory.clear()
    def IsFull(self):
        return len(self.memory) > self.maxMemory
    def getRandomBatch(self):
        """
        Get a random batch of states and Q-values from the replay-memory.
        You must call prepare_sampling_prob() before calling this function,
        which also sets the batch-size.

        The batch has been balanced so it contains states and Q-values
        that have both high and low estimation errors for the Q-values.
        This is done to both speed up and stabilize training of the
        Neural Network.
        """

        # Random index of states and Q-values in the replay-memory.
        # These have LOW estimation errors for the Q-values.
        # idx_lo = np.random.choice(self.idx_err_lo,
         #                         size=self.num_samples_err_lo,
         #                         replace=False)

        # Random index of states and Q-values in the replay-memory.
        # These have HIGH estimation errors for the Q-values.
        #idx_hi = np.random.choice(self.idx_err_hi,
       #                           size=self.num_samples_err_hi,
       #                           replace=False)

        # Combine the indices.
        #idx = np.concatenate((idx_lo, idx_hi))

        # Get the batches of states and Q-values.
        randomState = self.memory[random.randint(0,len(self.memory)-1)]
        return randomState
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

