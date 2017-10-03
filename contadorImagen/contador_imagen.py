import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot

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
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        #print('**resultado obtenido')
        #print(final_outputs)
        
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
#se duplica la cantidad de nodos del input original (784)
input_nodes = 1568
hidden_nodes = 400
output_nodes = 20
output_por_numero = 10

# learning rate
learning_rate = 0.6

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_decenas = open("mnist_train.csv", 'r')
training_decenas_list = training_decenas.readlines()
training_decenas.close()

training_unidades = open("mnist_train.csv", 'r')
training_unidades_list = training_unidades.readlines()
training_unidades.close()


# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 30

for e in range(epochs):
    # go through all records in the training data set
    numero_linea = 0
    for record in training_unidades_list:
        # split the record by the ',' commas
        all_values_unidades = training_unidades_list[numero_linea].split(',')
        #print('**entrenando')
        # scale and shift the inputs
        inputs_unidades = (numpy.asfarray(all_values_unidades[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets_unidades = numpy.zeros(output_por_numero) + 0.01
        #print('largo targets_unidades ')
        #print(len(targets_unidades))
       
        ##
        all_values_decenas = training_decenas_list[numero_linea].split(',')
        # scale and shift the inputs
        inputs_decenas = (numpy.asfarray(all_values_decenas[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets_decenas = numpy.zeros(output_por_numero) + 0.01
        #print('largo target_decenas ')
        #print(len(targets_decenas))
        
        
        # all_values_unidades[0] is the target label for this record
        numero_unidad = all_values_unidades[0]
        #print('entrada unidad: ' + numero_unidad)
        
        numero_decena = all_values_decenas[0]
        #print('entrada : ' + numero_decena + '' +numero_unidad)

        numero_siguiente = int(numero_decena + numero_unidad) + 1
        
        if numero_siguiente < 10:
            numero_siguiente = '0' + str(numero_siguiente)

        numero_siguiente = str(numero_siguiente)
            
        if numero_siguiente == '100':
            #print('numero_siguiente: 00')
            numero_unidad = 0
            numero_decena = 0
        else:
            #print('numero_siguiente: ' + numero_siguiente)
            numero_decena = int(numero_siguiente[0])
            numero_unidad = int(numero_siguiente[1])
        
        #print('*esperado/target*')
        #target para unidades
        targets_unidades[numero_unidad] = 0.99
        #print('targets_unidades:')
        #print(targets_unidades)

        # all_values_decenas[0] is the target label for this record
        #target para decenas
        targets_decenas[numero_decena] = 0.99
        #print('targets_decenas:')
        #print(targets_decenas)

        input_training = numpy.append(inputs_decenas, inputs_unidades)
        target_esperado = numpy.append(targets_decenas,targets_unidades)
        """
        print('largo input training')
        print(len(input_training))
        print('largo target_esperado')
        print(len(target_esperado))
        print(target_esperado)
        print('fin target_esperado')
        """
        numero_linea = numero_linea + 1
        n.train(input_training, target_esperado)
        pass
    pass


# load the mnist test data CSV file into a list
test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


# test the neural network

# scorecard for how well the network performs, initially empty
scorecard_decenas = []
scorecard_unidades = []

# go through all the records in the test data set
for record in test_data_list:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label_decenas = int(all_values[0])
    correct_label_unidades = int(all_values[0])
    print('entrada: '+str(correct_label_decenas) + str(correct_label_unidades))
    # scale and shift the inputs
    inputs_decenas = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    inputs_unidades = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(numpy.append(inputs_decenas, inputs_unidades))
    # the index of the highest value corresponds to the label
    label_decenas = numpy.argmax(outputs[:10])
    label_unidades = numpy.argmax(outputs[10:])

    print('resultado: '+ str(label_decenas)+str(label_unidades))
    # append correct or incorrect to list
    if (label_decenas == correct_label_decenas):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard_decenas.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard_decenas.append(0)
        pass
    if (label_unidades == correct_label_unidades):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard_unidades.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard_unidades.append(0)
        pass
    
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array_decenas = numpy.asarray(scorecard_decenas)
print ("performance decenas = ", (scorecard_array_decenas.sum() + 0.0) / (scorecard_array_decenas.size+ 0.0))

# calculate the performance score, the fraction of correct answers
scorecard_array_unidades = numpy.asarray(scorecard_decenas)
print ("performance unidades= ", (scorecard_array_unidades.sum()+ 0.0) / (scorecard_array_unidades.size+ 0.0))