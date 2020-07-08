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
        self.estimationErrors = np.zeros(shape=200000, dtype=np.float)
        self.errorThreshold = 0.1
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
        for k in reversed(range(len(self.memory))):
            # Get the data for the k'th state in the replay-memory.
            state =  self.memory[k]
            action = self.memory[k].action
            reward = self.memory[k].reward
            end_episode = self.memory[k].endEpisode
            if( reward >0):
                reward = reward
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
            self.estimationErrors[k] = abs(actionValue - self.memory[k].qValues[action])
        
            # Update the Q-value with the better estimate.
            self.memory[k].qValues[action] = actionValue
        #self.print_statistics()
    def PrintReplays(self):
        for current in range(len(self.memory)):
            currentState = self.memory[current]
            Utils.Logger.Logger.Log("Index" + str(current) +" action: " + str(currentState.action) + " reward: " + str(currentState.reward) + " end of episode:" + str(currentState.endEpisode))
    def ClearMemory(self):
        self.memory.clear()
    def IsFull(self):
        return len(self.memory) > self.maxMemory

    def prepare_sampling_prob(self, batch_size=128):
        """
        Prepare the probability distribution for random sampling of states
        and Q-values for use in training of the Neural Network.
        The probability distribution is just a simple binary split of the
        replay-memory based on the estimation errors of the Q-values.
        The idea is to create a batch of samples that are balanced somewhat
        evenly between Q-values that the Neural Network already knows how to
        estimate quite well because they have low estimation errors, and
        Q-values that are poorly estimated by the Neural Network because
        they have high estimation errors.
        
        The reason for this balancing of Q-values with high and low estimation
        errors, is that if we train the Neural Network mostly on data with
        high estimation errors, then it will tend to forget what it already
        knows and hence become over-fit so the training becomes unstable.
        """

        # Get the errors between the Q-values that were estimated using
        # the Neural Network, and the Q-values that were updated with the
        # reward that was actually observed when an action was taken.
        err = self.estimationErrors[0:len(self.memory)]

        # Create an index of the estimation errors that are low.
        idx = err<self.errorThreshold
        self.idx_err_lo = np.squeeze(np.where(idx))

        # Create an index of the estimation errors that are high.
        self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))

        # Probability of sampling Q-values with high estimation errors.
        # This is either set to the fraction of the replay-memory that
        # has high estimation errors - or it is set to 0.5. So at least
        # half of the batch has high estimation errors.
        prob_err_hi = len(self.idx_err_hi) / len(self.memory)
        prob_err_hi = max(prob_err_hi, 0.5)

        # Number of samples in a batch that have high estimation errors.
        self.num_samples_err_hi = int(prob_err_hi * batch_size)

        # Number of samples in a batch that have low estimation errors.
        self.num_samples_err_lo = batch_size - self.num_samples_err_hi

    def random_batch(self):
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
        idx_lo = np.random.choice(self.idx_err_lo,
                                  size=self.num_samples_err_lo,
                                  replace=False)

        # Random index of states and Q-values in the replay-memory.
        # These have HIGH estimation errors for the Q-values.
        idx_hi = np.random.choice(self.idx_err_hi,
                                  size=self.num_samples_err_hi,
                                  replace=False)

        # Combine the indices.
        idx = np.concatenate((idx_lo, idx_hi))

        # Get the batches of states and Q-values.
        states_batch = self.memory[idx].state
        q_values_batch = self.memory[idx].qValues

        return states_batch, q_values_batch

    def all_batches(self, batch_size=128):
        """
        Iterator for all the states and Q-values in the replay-memory.
        It returns the indices for the beginning and end, as well as
        a progress-counter between 0.0 and 1.0.
        
        This function is not currently being used except by the function
        estimate_all_q_values() below. These two functions are merely
        included to make it easier for you to experiment with the code
        by showing you an easy and efficient way to loop over all the
        data in the replay-memory.
        """

        # Start index for the current batch.
        begin = 0

        # Repeat until all batches have been processed.
        while begin < self.num_used:
            # End index for the current batch.
            end = begin + batch_size

            # Ensure the batch does not exceed the used replay-memory.
            if end > self.num_used:
                end = self.num_used

            # Progress counter.
            progress = end / self.num_used

            # Yield the batch indices and completion-counter.
            yield begin, end, progress

            # Set the start-index for the next batch to the end of this batch.
            begin = end

    def estimate_all_q_values(self, model):
        """
        Estimate all Q-values for the states in the replay-memory
        using the model / Neural Network.
        Note that this function is not currently being used. It is provided
        to make it easier for you to experiment with this code, by showing
        you an efficient way to iterate over all the states and Q-values.
        :param model:
            Instance of the NeuralNetwork-class.
        """

        print("Re-calculating all Q-values in replay memory ...")

        # Process the entire replay-memory in batches.
        for begin, end, progress in self.all_batches():
            # Print progress.
            msg = "\tProgress: {0:.0%}"
            msg = msg.format(progress)
            print_progress(msg)

            # Get the states for the current batch.
            states = self.states[begin:end]

            # Calculate the Q-values using the Neural Network
            # and update the replay-memory.
            self.q_values[begin:end] = model.get_q_values(states=states)

        # Newline.
        print()

    def print_statistics(self):
        """Print statistics for the contents of the replay-memory."""

        print("Replay-memory statistics:")

        # Print statistics for the Q-values before they were updated
        # in update_all_q_values().
        msg = "\tQ-values Before, Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values_old),
                         np.mean(self.q_values_old),
                         np.max(self.q_values_old)))

        # Print statistics for the Q-values after they were updated
        # in update_all_q_values().
        msg = "\tQ-values After,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(self.q_values),
                         np.mean(self.q_values),
                         np.max(self.q_values)))

        # Print statistics for the difference in Q-values before and
        # after the update in update_all_q_values().
        q_dif = self.q_values - self.q_values_old
        msg = "\tQ-values Diff.,  Min: {0:5.2f}, Mean: {1:5.2f}, Max: {2:5.2f}"
        print(msg.format(np.min(q_dif),
                         np.mean(q_dif),
                         np.max(q_dif)))

        # Print statistics for the number of large estimation errors.
        # Don't use the estimation error for the last state in the memory,
        # because its Q-values have not been updated.
        err = self.estimation_errors[:-1]
        err_count = np.count_nonzero(err > self.error_threshold)
        msg = "\tNumber of large errors > {0}: {1} / {2} ({3:.1%})"
        print(msg.format(self.error_threshold, err_count,
                         self.num_used, err_count / self.num_used))

        # How much of the replay-memory is used by states with end_life.
        end_life_pct = np.count_nonzero(self.end_life) / self.num_used

        # How much of the replay-memory is used by states with end_episode.
        end_episode_pct = np.count_nonzero(self.end_episode) / self.num_used

        # How much of the replay-memory is used by states with non-zero reward.
        reward_nonzero_pct = np.count_nonzero(self.rewards) / self.num_used

        # Print those statistics.
        msg = "\tend_life: {0:.1%}, end_episode: {1:.1%}, reward non-zero: {2:.1%}"
        print(msg.format(end_life_pct, end_episode_pct, reward_nonzero_pct))

########################################################################
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

