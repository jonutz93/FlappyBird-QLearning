import tensorflow as tf
import numpy
import Utils.Logger
import random
import Constants
class Brain(object):
    """description of class"""
    def __init__(self,id):
        #parameters
        self.n_input = Constants.inputLayer
        self.n_hidden = Constants.hiddenLyaer
        self.n_output = Constants.outputLayer
        self.learning_rate = Constants.learningRate
        self.epochs = Constants.epochs
        self.setup()
        self.id = id
    def setup(self):
        #placeholders
        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)
        #weights
        w1 = tf.Variable(tf.random_uniform([self.n_input, self.n_hidden], -1.0, 1.0))
        w2 = tf.Variable(tf.random_uniform([self.n_hidden, self.n_output], -1.0, 1.0))
        b1 = tf.Variable(tf.zeros([self.n_hidden]))
        b2 = tf.Variable(tf.zeros([self.n_output]))
             
        self.weight1 =numpy.random.random((self.n_input, self.n_hidden)).astype(numpy.float32)
        self.weight2 = numpy.random.random((self.n_hidden, self.n_output)).astype(numpy.float32)
        self.bias1 = numpy.random.random((self.n_hidden))
        self.bias2=numpy.random.random(self.n_output)
        #Weights
        self.W1 = tf.Variable(self.weight1)
        self.W2 = tf.Variable(self.weight2)
        
        # Bias
        self.b1 = tf.Variable(tf.zeros([self.n_hidden]) , name="B1")
        self.b2 = tf.Variable(tf.zeros([self.n_output]) , name="B2")

        #Layers
        self.L2 = tf.sigmoid(tf.matmul(self.X, self.W1) + self.b1)
        self.hy = tf.sigmoid(tf.matmul(self.L2, self.W2) + self.b2)
        
        #not sure what this does yet
        self.cost = tf.reduce_mean(-self.Y*tf.log(self.hy) - (1-self.Y)*tf.log(1-self.hy))
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
        #weights
        self.W1_placeholder = tf.placeholder(tf.float32, shape=[self.n_input, self.n_hidden])
        self.W2_placeholder = tf.placeholder(tf.float32, shape=[self.n_hidden, self.n_output])
        self.W1_assign = tf.assign(self.W1, self.W1_placeholder)
        self.W2_assign = tf.assign(self.W2, self.W2_placeholder)
        #bias
        self.B1_placeholder = tf.placeholder(tf.float32, shape=[self.n_hidden])
        self.B2_placeholder = tf.placeholder(tf.float32, shape=[self.n_output])
        self.b1_assign = tf.assign(self.b1, self.B1_placeholder)
        self.b2_assign = tf.assign(self.b2, self.B2_placeholder)
        self.init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(self.init)
    def updateWeights(self,weights1,weights2,bias1,bias2):
        #internal weights. Are generated random for now. I hjave to generated them here otherwise are generated at each iteration we have new numbers.
        #last colomn represents the bias
        self.weight1 = weights1
        self.weight2 = weights2
        self.bias1 = bias1
        self.bias2 =bias2
        self.session.run([self.W1_assign, self.W2_assign],feed_dict={self.W1_placeholder:weights1, self.W2_placeholder:weights2})
        self.session.run([self.b1_assign, self.b2_assign],feed_dict={self.B1_placeholder:bias1, self.B2_placeholder:bias2})
        Logger.Logger.Log("Bird id is "+str(self.id))
        self.printWeights()
    def randomWeights(self):
        #internal weights. Are generated random for now. I have to generated them here otherwise are generated at each iteration we have new numbers.
        #last colomn represents the bias
        weight1 =numpy.random.uniform(low=-1.,high=1.,size=(self.n_input, self.n_hidden)).astype(numpy.float32)
        weight2 = numpy.random.uniform(low=-1.,high=1.,size=(self.n_hidden, self.n_output)).astype(numpy.float32)
        bias1 = numpy.random.uniform(low=-1.,high=1.,size=(self.n_hidden))
        bias2 = numpy.random.uniform(low=-1.,high=1.,size=(self.n_output))
        self.updateWeights(weight1,weight2,bias1,bias2)
    def getWeights(self):
        return {"w1":self.weight1,"w2":self.weight2,"bias1":self.bias1,"bias2":self.bias2}
    def updateWeightsJson(self,newWeights):
        self.updateWeights(newWeights["w1"],newWeights["w2"],newWeights["bias1"],newWeights["bias2"])
    def copyBrain(self):
        #untested
        newBrain = Brain()
        newBrain.setup()
        newBrain.updateWeights(self.Weights1,self.Weights2,self.bias1,self.bias2)
        return newBrain
    def printWeights(self):
       return 0
       Logger.Logger.Log("W1")
       Logger.Logger.Log(numpy.array2string(self.W1.eval(self.session)))
       Logger.Logger.Log("W2")
       Logger.Logger.Log(numpy.array2string(self.W2.eval(self.session)))
       Logger.Logger.Log("bias1")
       Logger.Logger.Log(numpy.array2string(self.b1.eval(self.session)))
       Logger.Logger.Log("bias2")
       Logger.Logger.Log(numpy.array2string(self.b2.eval(self.session)))

    def CalculateQValues(self, input):
        legitAnswer=self.session.run([self.net], feed_dict={self.X: input})
        return answer
    def mutate_w_with_percent_change(self,p, add_sub_rand=True):
        #considering its 2d array
        new_p = []
        for i in p:
            row = []
            for j in i:
                temp = j
                delta = numpy.random.random_sample() + 0.5
                if numpy.random.random_sample() > 0.5:
                    temp = temp * delta
                if add_sub_rand == True:
                    if numpy.random.random_sample() > 0.5:
                        if numpy.random.random_sample() > 0.5:
                            temp = temp - numpy.random.random_sample()
                        else:
                            temp = temp + numpy.random.random_sample()
                row.append(temp)
            new_p.append(row)
        return new_p
    def mutate_b_with_percent_change(self,p, add_sub_rand=True):
        #considering its 1d array
        new_p = []
        for i in p:
            temp = i
            delta = numpy.random.random_sample() + 0.5
            if numpy.random.random_sample() > 0.5:
                temp = temp * delta
            if add_sub_rand == True:
                if numpy.random.random_sample() > 0.5:
                    if numpy.random.random_sample() > 0.5:
                        temp = temp - numpy.random.random_sample()
                    else:
                        temp = temp + numpy.random.random_sample()
            new_p.append(temp)
        return new_p
    def mutate(self,prob):
        for ix,iy in numpy.ndindex(self.weight1.shape):
            self.weight1[ix,iy] = self.mutateValue(self.weight1[ix,iy],prob)
        for ix,iy in numpy.ndindex(self.weight2.shape):
            self.weight2[ix,iy] = self.mutateValue(self.weight2[ix,iy],prob)
        for ix in numpy.ndindex(self.bias1.shape):
           self.bias1[ix] =  self.mutateValue(self.bias1[ix],prob)
        for ix in numpy.ndindex(self.bias2.shape):
           self.bias2[ix] = self.mutateValue(self.bias2[ix],prob)
        self.updateWeights(self.weight1,self.weight2,self.bias1,self.bias2)
    def mutateValue(self,value,prob):
        if (random.uniform(0, 1) < prob):
            offset = random.uniform(-1, 1) * 0.5;
            newValue = value + offset
            return newValue
        return value
    def mutateValue2(self,value,prob):
        #another try
        if (random.uniform(0, 1) < prob):
            mutateFactor = 1 + ((random.uniform(0, 1) - 0.5) * 3 + (random.uniform(0, 1)- 0.5))
            value *= mutateFactor;
        return value
    def crossOver(self,brain):
        w1_ = self.cross_over(self.weight1,self.weight2,self.bias1,self.bias2,brain.weight1,brain.weight2,brain.bias1,brain.bias2)
        w2_ = self.mutate_w_with_percent_change(self.weight2)
        b1_ = self.mutate_b_with_percent_change(self.bias1)
        b2_ = self.mutate_b_with_percent_change(self.bias2)
        self.updateWeights(w1_,w2_,b1,b2_)
    def crossOver(self,brain1,brain2):
        output = self.cross_over(brain1.weight1,brain1.weight2,brain1.bias1,brain1.bias2,brain2.weight1,brain2.weight2,brain2.bias1,brain2.bias2)
        w1_ = output[0]
        w2_ = output[1]
        b1_ = output[2]
        b2_ = output[3]
        self.updateWeights(w1_,w2_,b1_,b2_)
    def cross_over(self,w11, w12, b11, b12, w21, w22, b21, b22):
        new_w1 = []
        for i in range(len(w11)):
            row = []
            for j in range(len(w11[0])):
                if numpy.random.random_sample() > 0.5:
                    row.append(w11[i][j])
                else:
                    row.append(w21[i][j])
            new_w1.append(row)
        new_w2 = []
        for i in range(len(w12)):
            row = []
            for j in range(len(w12[0])):
                if numpy.random.random_sample() > 0.5:
                    row.append(w12[i][j])
                else:
                    row.append(w22[i][j])
            new_w2.append(row)
        new_b1 = []
        for i in range(len(b11)):
            if numpy.random.random_sample() > 0.5:
                new_b1.append(b11[i])
            else:
                new_b1.append(b21[i])

        new_b2 = []
        for i in range(len(b12)):
            if numpy.random.random_sample() > 0.5:
                new_b2.append(b12[i])
            else:
                new_b2.append(b22[i])

        return (new_w1, new_w2, new_b1, new_b2)
            #var index = Math.floor(Math.random()*this.code.length);
            #var upOrDown = Math.random()
 