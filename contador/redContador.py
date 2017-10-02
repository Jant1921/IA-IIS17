
# coding: utf-8

# In[30]:


import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window



# In[31]:


# neural network class definition
class neuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        print('resultado training')
        print(final_inputs)
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    # number of input, hidden and output nodes
# 3 nodos de entrada por que usamos 3 cifras para representar los numeros
input_nodes = 3
hidden_nodes = 100
#3 nodos de salida por que usamos 3 cifras para representar los numeros
output_nodes = 3

# learning rate lo podemos variar segun la cantidad de entrenamiento que hagamos
learning_rate = 0.7

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)


# In[32]:


# load the mnist training data CSV file into a list
training_data_file = open("entradas.csv", 'r') # csv que tiene tanto el resultado correcto, como el input en binario ej: 0 0 1  0 0 0
training_data_list = training_data_file.readlines()
training_data_file.close()


# In[33]:


# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 2000

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        print('***entrenamiento***\n inputs:')
        print(all_values)
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[3:]) )
        inputs = map(int,inputs)
        
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) 
        # all_values[0] is the target label for this record
        targets = (numpy.asfarray(all_values[:3]))
        targets= map(int, targets)
        print('inputs')
        print(inputs)
        print('targets')
        print(targets)
        n.train(inputs, targets)
        pass
    pass


# In[34]:


# load the mnist test data CSV file into a list
test_data_file = open("entradas.csv", 'r') # csv que tiene tanto el resultado correcto, como el input en binario ej: 0 0 1  0 0 0
test_data_list = test_data_file.readlines()
test_data_file.close()


# In[35]:


# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = all_values[:3]
   
    print('\nInicio de prueba')
    

    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[3:]) ) 
    

    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    #label = numpy.argmax(outputs)
    label = [int(round(numb, 0)) for numb in outputs]
    print("Inputs = ", inputs)
    print("Valor esperado", correct_label)
    print("Resultado", label)
    print ("Outputs de la red = ", outputs)



# In[ ]:




