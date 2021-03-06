import tensorflow as tf
import numpy as np
import Utils.Logger
import random
import Constants
import os
class LinearControlSignal:
    """
    A control signal that changes linearly over time.

    This is used to change e.g. the learning-rate for the optimizer
    of the Neural Network, as well as other parameters.
    
    TensorFlow has functionality for doing this, but it uses the
    global_step counter inside the TensorFlow graph, while we
    want the control signals to use a state-counter for the
    game-environment. So it is easier to make this in Python.
    """

    def __init__(self, start_value, end_value, num_iterations, repeat=False):
        """
        Create a new object.

        :param start_value:
            Start-value for the control signal.

        :param end_value:
            End-value for the control signal.

        :param num_iterations:
            Number of iterations it takes to reach the end_value
            from the start_value.

        :param repeat:
            Boolean whether to reset the control signal back to the start_value
            after the end_value has been reached.
        """

        # Store arguments in this object.
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = num_iterations
        self.repeat = repeat

        # Calculate the linear coefficient.
        self._coefficient = (end_value - start_value) / num_iterations

    def get_value(self, iteration):
        """Get the value of the control signal for the given iteration."""

        if self.repeat:
            iteration %= self.num_iterations

        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value

        return value

########################################################################

class EpsilonGreedy:
    """
    The epsilon-greedy policy either takes a random action with
    probability epsilon, or it takes the action for the highest
    Q-value.
    
    If epsilon is 1.0 then the actions are always random.
    If epsilon is 0.0 then the actions are always argmax for the Q-values.

    Epsilon is typically decreased linearly from 1.0 to 0.1
    and this is also implemented in this class.

    During testing, epsilon is usually chosen lower, e.g. 0.05 or 0.01
    """

    def __init__(self, num_actions,
                 epsilon_testing=0.05,
                 num_iterations=1e6,
                 start_value=1.0, end_value=0.1,
                 repeat=False):
        """
        
        :param num_actions:
            Number of possible actions in the game-environment.

        :param epsilon_testing:
            Epsilon-value when testing.

        :param num_iterations:
            Number of training iterations required to linearly
            decrease epsilon from start_value to end_value.
            
        :param start_value:
            Starting value for linearly decreasing epsilon.

        :param end_value:
            Ending value for linearly decreasing epsilon.

        :param repeat:
            Boolean whether to repeat and restart the linear decrease
            when the end_value is reached, or only do it once and then
            output the end_value forever after.
        """

        # Store parameters.
        self.num_actions = num_actions
        self.epsilon_testing = epsilon_testing

        # Create a control signal for linearly decreasing epsilon.
        self.epsilon_linear = LinearControlSignal(num_iterations=num_iterations,
                                                  start_value=start_value,
                                                  end_value=end_value,
                                                  repeat=repeat)

    def get_epsilon(self, iteration, training):
        """
        Return the epsilon for the given iteration.
        If training==True then epsilon is linearly decreased,
        otherwise epsilon is a fixed number.
        """

        if training:
            epsilon = self.epsilon_linear.get_value(iteration=iteration)
        else:
            epsilon = self.epsilon_testing

        return epsilon

    def get_action(self, q_values, iteration, training):
        """
        Use the epsilon-greedy policy to select an action.
        
        :param q_values:
            These are the Q-values that are estimated by the Neural Network
            for the current state of the game-environment.
         
        :param iteration:
            This is an iteration counter. Here we use the number of states
            that has been processed in the game-environment.

        :param training:
            Boolean whether we are training or testing the
            Reinforcement Learning agent.

        :return:
            action (integer), epsilon (float)
        """

        epsilon = self.get_epsilon(iteration=iteration, training=training)

        # With probability epsilon.
        if np.random.random() < epsilon:
            # Select a random action.
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            # Otherwise select the action that has the highest Q-value.
            action = np.argmax(q_values)

        return action, epsilon

########################################################################

class Brain(object):
    """description of class"""
    def __init__(self,id, state_height, state_width):
        #parameters
        self.learning_rate = Constants.learningRate
        self.epochs = Constants.epochs
        self.setup(state_height, state_width)
        self.id = id
    def setup(self, state_height, state_width):
        # Shape of the state-array.
        state_shape = [state_height, state_width, 1]
        num_actions = Constants.outputLayer
        # Path for saving/restoring checkpoints.
        self.checkpoint_path = "./saveData"
        # Placeholder variable for inputting states into the Neural Network.
        # A state is a multi-dimensional array holding image-frames from
        # the game-environment.
        self.x = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name='x')

        # Placeholder variable for inputting the learning-rate to the optimizer.
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        # Placeholder variable for inputting the target Q-values
        # that we want the Neural Network to be able to estimate.
        self.q_values_new = tf.placeholder(tf.float32,
                                           shape=[None, num_actions],
                                           name='q_values_new')

        # This is a hack that allows us to save/load the counter for
        # the number of states processed in the game-environment.
        # We will keep it as a variable in the TensorFlow-graph
        # even though it will not actually be used by TensorFlow.
        self.count_states = tf.Variable(initial_value=0,
                                        trainable=False, dtype=tf.int64,
                                        name='count_states')

        # Similarly, this is the counter for the number of episodes.
        self.count_episodes = tf.Variable(initial_value=0,
                                          trainable=False, dtype=tf.int64,
                                          name='count_episodes')

        # TensorFlow operation for increasing count_states.
        self.count_states_increase = tf.assign(self.count_states,
                                               self.count_states + 1)

        # TensorFlow operation for increasing count_episodes.
        self.count_episodes_increase = tf.assign(self.count_episodes,
                                                 self.count_episodes + 1)

        # The Neural Network will be constructed in the following.
        # Note that the architecture of this Neural Network is very
        # different from that used in the original DeepMind papers,
        # which was something like this:
        # Input image:      84 x 84 x 4 (4 gray-scale images of 84 x 84 pixels).
        # Conv layer 1:     16 filters 8 x 8, stride 4, relu.
        # Conv layer 2:     32 filters 4 x 4, stride 2, relu.
        # Fully-conn. 1:    256 units, relu. (Sometimes 512 units).
        # Fully-conn. 2:    num-action units, linear.

        # The DeepMind architecture does a very aggressive downsampling of
        # the input images so they are about 10 x 10 pixels after the final
        # convolutional layer. I found that this resulted in significantly
        # distorted Q-values when using the training method further below.
        # The reason DeepMind could get it working was perhaps that they
        # used a very large replay memory (5x as big as here), and a single
        # optimization iteration was performed after each step of the game,
        # and some more tricks.

        # Initializer for the layers in the Neural Network.
        # If you change the architecture of the network, particularly
        # if you add or remove layers, then you may have to change
        # the stddev-parameter here. The initial weights must result
        # in the Neural Network outputting Q-values that are very close
        # to zero - but the network weights must not be too low either
        # because it will make it hard to train the network.
        # You can experiment with values between 1e-2 and 1e-3.
        init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

        # This builds the Neural Network using the tf.layers API,
        # which is very verbose and inelegant, but should work for everyone.

        # Padding used for the convolutional layers.
        padding = 'SAME'

        # Activation function for all convolutional and fully-connected
        # layers, except the last.
        activation = tf.nn.relu

        # Reference to the lastly added layer of the Neural Network.
        # This makes it easy to add or remove layers.
        net = self.x

        # First convolutional layer.
        #net = tf.layers.conv2d(inputs=net, name='layer_conv1',
        #                       filters=16, kernel_size=3, strides=2,
        #                       padding=padding,
        #                       kernel_initializer=init, activation=activation)
        #
        ## Second convolutional layer.
        #net = tf.layers.conv2d(inputs=net, name='layer_conv2',
        #                       filters=32, kernel_size=3, strides=2,
        #                       padding=padding,
        #                       kernel_initializer=init, activation=activation)
        #
        ## Third convolutional layer.
        #net = tf.layers.conv2d(inputs=net, name='layer_conv3',
        #                       filters=64, kernel_size=3, strides=1,
        #                       padding=padding,
        #                       kernel_initializer=init, activation=activation)
        
        # Flatten output of the last convolutional layer so it can
        # be input to a fully-connected (aka. dense) layer.
        # TODO: For some bizarre reason, this function is not yet in tf.layers
        # TODO: net = tf.layers.flatten(net)
        net = tf.contrib.layers.flatten(net)

        # First fully-connected (aka. dense) layer.
        net = tf.layers.dense(inputs=net, name='layer_fc1', units=1024,
                              kernel_initializer=init, activation=activation)

        # Second fully-connected layer.
        net = tf.layers.dense(inputs=net, name='layer_fc2', units=1024,
                              kernel_initializer=init, activation=activation)

        # Third fully-connected layer.
        net = tf.layers.dense(inputs=net, name='layer_fc3', units=1024,
                              kernel_initializer=init, activation=activation)

        # Fourth fully-connected layer.
        net = tf.layers.dense(inputs=net, name='layer_fc4', units=1024,
                              kernel_initializer=init, activation=activation)

        # Final fully-connected layer.
        net = tf.layers.dense(inputs=net, name='layer_fc_out', units=num_actions,
                              kernel_initializer=init, activation=None)

        # The output of the Neural Network is the estimated Q-values
        # for each possible action in the game-environment.
        self.q_values = net

        # TensorFlow has a built-in loss-function for doing regression:
        # self.loss = tf.nn.l2_loss(self.q_values - self.q_values_new)
        # But it uses tf.reduce_sum() rather than tf.reduce_mean()
        # which is used by PrettyTensor. This means the scale of the
        # gradient is different and hence the hyper-parameters
        # would have to be re-tuned, because they were tuned for
        # the original version of this tutorial using PrettyTensor.
        # So instead we calculate the L2-loss similarly to how it is
        # done in PrettyTensor.
        squared_error = tf.square(self.q_values - self.q_values_new)
        sum_squared_error = tf.reduce_sum(squared_error, axis=1)
        self.loss = tf.reduce_mean(sum_squared_error)

        # Optimizer used for minimizing the loss-function.
        # Note the learning-rate is a placeholder variable so we can
        # lower the learning-rate as optimization progresses.
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Used for saving and loading checkpoints.
        self.saver = tf.train.Saver()

        # Create a new TensorFlow session so we can run the Neural Network.
        self.session = tf.Session()

        # Load the most recent checkpoint if it exists,
        # otherwise initialize all the variables in the TensorFlow graph.
        self.loadCheckpoint()

    def loadCheckpoint(self):
        try:
            print("Trying to restore last checkpoint ...")
        
            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=self.checkpoint_path)
        
            # Try and load the data in the checkpoint.
            self.saver.restore(self.session, save_path=last_chk_path)
        
            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint from:", self.checkpoint_path)
            print("Initializing variables instead.")
            self.session.run(tf.global_variables_initializer())

    def SaveCheckpoint(self):
        """Save all variables of the TensorFlow graph to a checkpoint."""

        self.saver.save(self.session,
                        save_path=self.checkpoint_path+"/data.ckpt")

        print("Saved checkpoint.")

    def get_q_values(self, states):
        """
        Calculate and return the estimated Q-values for the given states.

        A single state contains two images (or channels): The most recent
        image-frame from the game-environment, and a motion-tracing image.
        See the MotionTracer-class for details.

        The input to this function is an array of such states which allows
        for batch-processing of the states. So the input is a 4-dim
        array with shape: [batch, height, width, state_channels].
        
        The output of this function is an array of Q-value-arrays.
        There is a Q-value for each possible action in the game-environment.
        So the output is a 2-dim array with shape: [batch, num_actions]
        """

        # Create a feed-dict for inputting the states to the Neural Network.
        newState = np.array
        feed_dict = {self.x: states}

        # Use TensorFlow to calculate the estimated Q-values for these states.
        values = self.session.run(self.q_values, feed_dict=feed_dict)
        return values

    def optimize(self, replay_memory, min_epochs=1.0, max_epochs=10,
                 batch_size=128, loss_limit=0.015,
                 learning_rate=1e-3):
        """
        Optimize the Neural Network by sampling states and Q-values
        from the replay-memory.

        The original DeepMind paper performed one optimization iteration
        after processing each new state of the game-environment. This is
        an un-natural way of doing optimization of Neural Networks.

        So instead we perform a full optimization run every time the
        Replay Memory is full (or it is filled to the desired fraction).
        This also gives more efficient use of a GPU for the optimization.

        The problem is that this may over-fit the Neural Network to whatever
        is in the replay-memory. So we use several tricks to try and adapt
        the number of optimization iterations.

        :param min_epochs:
            Minimum number of optimization epochs. One epoch corresponds
            to the replay-memory being used once. However, as the batches
            are sampled randomly and biased somewhat, we may not use the
            whole replay-memory. This number is just a convenient measure.

        :param max_epochs:
            Maximum number of optimization epochs.

        :param batch_size:
            Size of each random batch sampled from the replay-memory.

        :param loss_limit:
            Optimization continues until the average loss-value of the
            last 100 batches is below this value (or max_epochs is reached).

        :param learning_rate:
            Learning-rate to use for the optimizer.
        """

        print("Optimizing Neural Network to better estimate Q-values ...")
        print("\tLearning-rate: {0:.1e}".format(learning_rate))
        print("\tLoss-limit: {0:.3f}".format(loss_limit))
        print("\tMax epochs: {0:.1f}".format(max_epochs))

        # Prepare the probability distribution for sampling the replay-memory.
        #replay_memory.prepare_sampling_prob(batch_size=batch_size)

        # Number of optimization iterations corresponding to one epoch.
        iterations_per_epoch = batch_size

        # Minimum number of iterations to perform.
        min_iterations = int(iterations_per_epoch * min_epochs)

        # Maximum number of iterations to perform.
        max_iterations = 1 #int(len(replay_memory.memory) /10)

        # Buffer for storing the loss-values of the most recent batches.
        loss_history = np.zeros(100, dtype=float)

        for i in range(len(replay_memory.memory)):
            # Randomly sample a batch of states and target Q-values
            # from the replay-memory. These are the Q-values that we
            # want the Neural Network to be able to estimate.
            randomState = replay_memory.memory[i]
            state_batch= randomState.state
            q_values_batch = randomState.qValues
            # Create a feed-dict for inputting the data to the TensorFlow graph.
            # Note that the learning-rate is also in this feed-dict.
            feed_dict = {self.x: [state_batch],
                         self.q_values_new: [q_values_batch],
                         self.learning_rate: learning_rate}

            # Perform one optimization step and get the loss-value.
            loss_val, _ = self.session.run([self.loss, self.optimizer],
                                           feed_dict=feed_dict)

            # Shift the loss-history and assign the new value.
            # This causes the loss-history to only hold the most recent values.
            loss_history = np.roll(loss_history, 1)
            loss_history[0] = loss_val

            # Calculate the average loss for the previous batches.
            loss_mean = np.mean(loss_history)

            # Print status.
            pct_epoch = i / iterations_per_epoch
            msg = "\tIteration: {0} ({1:.2f} epoch), Batch loss: {2:.4f}, Mean loss: {3:.4f}"
            msg = msg.format(i, pct_epoch, loss_val, loss_mean)
            print(msg)

            # Stop the optimization if we have performed the required number
            # of iterations and the loss-value is sufficiently low.
            if i > min_iterations and loss_mean < loss_limit:
                break

        # Print newline.
        self.SaveCheckpoint()

    def get_count_states(self):
        """
        Get the number of states that has been processed in the game-environment.
        This is not used by the TensorFlow graph. It is just a hack to save and
        reload the counter along with the checkpoint-file.
        """
        return self.session.run(self.count_states)

    def get_count_episodes(self):
        """
        Get the number of episodes that has been processed in the game-environment.
        """
        return self.session.run(self.count_episodes)

    def increase_count_states(self):
        """
        Increase the number of states that has been processed
        in the game-environment.
        """
        return self.session.run(self.count_states_increase)