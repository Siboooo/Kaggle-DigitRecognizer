import numpy
import scipy.special
import scipy.ndimage
import pandas

class NeuralNetwork:

    #initialisation
    def __init__(self, input_nodes_num, hidden_nodes_num, output_nodes_num, learning_rate):
        self.inputNum = input_nodes_num
        self.hiddenNum = hidden_nodes_num
        self.outputNum = output_nodes_num
        self.learningRate = learning_rate

        #set up initial weights
        self.weightsIH = numpy.random.normal(0.0, pow(self.hiddenNum, -0.5), (self.hiddenNum, self.inputNum))
        self.weightsHO = numpy.random.normal(0.0, pow(self.outputNum, -0.5), (self.outputNum, self.hiddenNum))

        #set up Activation Function
        self.activationFun = lambda x: scipy.special.expit(x)
        pass

    #train the NN
    def train(self, input_nodes, target_nodes):
        #correct answer
        target = numpy.array(target_nodes, ndmin = 2).T

        #forward propagation
        input = numpy.array(input_nodes, ndmin = 2).T
        hidden = self.activationFun(numpy.dot(self.weightsIH, input))
        output = self.activationFun(numpy.dot(self.weightsHO, hidden))

        #backward propagation
        oError = target - output
        hError = numpy.dot(self.weightsHO.T, oError)

        #update weights
        self.weightsHO += self.learningRate * numpy.dot((oError * output * (1.0 - output)), numpy.transpose(hidden))
        self.weightsIH += self.learningRate * numpy.dot((hError * hidden * (1.0 - hidden)), numpy.transpose(input))
        pass

    #make prediction
    def query(self, input_nodes):
        input = numpy.array(input_nodes, ndmin = 2).T
        hidden = self.activationFun(numpy.dot(self.weightsIH, input))
        output = self.activationFun(numpy.dot(self.weightsHO, hidden))
        return output

    pass

#basic info
inputNodes = 784
hiddenNodes = 200
outputNodes = 10
learningRate = 0.01

n = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)

#import training data
#try to train a small dataset first
#trainingDataFile = open("mnist_train_100.csv", "r")
trainingDataFile = open("train.csv", "r")
trainingData = trainingDataFile.readlines()
trainingDataFile.close()

#remove the label
trainingData = trainingData[1:]

#epochs
for e in range(10):
    for record in trainingData:
        values = record.split(",")
        #scale & shift input features
        input = (numpy.asfarray(values[1:])/255.0*0.99) + 0.01
        #create the target for the training
        targets = numpy.zeros(outputNodes) + 0.01
        targets[int(values[0])] = 0.99
        n.train(input, targets)

        #Data augmentation
        #rotate 10 degree
        inputP = scipy.ndimage.interpolation.rotate(input.reshape(28, 28), 10, reshape=False, cval=0.01)
        n.train(inputP.reshape(784), targets)

        #rotate -10 degree
        inputM = scipy.ndimage.interpolation.rotate(input.reshape(28, 28), -10, reshape=False, cval=0.01)
        n.train(inputM.reshape(784), targets)

#inport test data
#try to recognize a small test set first
#testDataFile = open("mnist_test_10.csv", "r")
testDataFile = open("test.csv", "r")
testData = testDataFile.readlines()
testDataFile.close()
#remove the label
testData = testData[1:]

#create data frame for the final result
result = pandas.DataFrame(columns=["Imageid", "Label"])
i = 1
for record in testData:
    values = record.split(",")
    input = (numpy.asfarray(values)/255.0*0.99) + 0.01
    output = n.query(input)
    prediction = numpy.argmax(output)

    temp = pandas.Series({"Imageid":i,"Label":prediction})
    result = result.append(temp, ignore_index=True)
    i += 1
#create the csv file
result.to_csv("sub.csv", index=False)
