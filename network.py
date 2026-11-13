from neuron import neuron
from layer import layer
import random

class network:
    def __init__(self, layers):
        self.layers = layers
        self.predictions = []
        
    def feedforward(self,targets, isTraining):
        for i in range(1,len(self.layers)):
            self.layers[i].feedforward(self.layers[i-1].outputs)
            print(f"layer {i} values: {self.layers[i].outputs}")
        self.predictions = self.layers[-1].outputs
        if isTraining:
            self.backpropagation(targets)
    
    def backpropagation(self,targets):
        for i in range(len(self.layers)-1,0,-1):
            print(f"in backpropagation layer {i}")
            if i == len(self.layers)-1:             
                self.layers[i].backpropagation(targets, self.layers[i-1].outputs)
            else:
                self.layers[i].backpropagation(None ,self.layers[i-1].outputs, self.layers[i+1])
            
    def train_step(self, inputs, targets):
        for i in range(len(inputs)):
            self.layers[0].output = inputs[i]
            self.feedforward(targets[i],True)

                    

if __name__ == "__main__":
    
    #initializing training set
    training_set = [[random.uniform(0,1) for _ in range(2)] for _ in range(1)]
    training_set_targets = [[random.uniform(0,1) for _ in range(2)] for _ in range(1)]
    
    
    #initializing input layer
    input_layer = layer()
    
    #initializing hidden layer
    hidden_neuron_1 = neuron([random.uniform(-1,1) for _ in range(2)])
    hidden_neuron_2 = neuron([random.uniform(-1,1) for _ in range(2)])
    hidden_neuron_3 = neuron([random.uniform(-1,1) for _ in range(2)])
    hidden_neuron_4 = neuron([random.uniform(-1,1) for _ in range(2)])
    
    hidden_layer = layer([hidden_neuron_1,hidden_neuron_2,hidden_neuron_3, hidden_neuron_4])
    
    #initializing output layer
    output_neuron_1 = neuron([random.uniform(-1,1) for _ in range(4)])
    output_neuron_2 = neuron([random.uniform(-1,1) for _ in range(4)])
    
    output_layer = layer([output_neuron_1,output_neuron_2])
    
    my_network = network([input_layer,hidden_layer,output_layer])
    
    for epoch in range(1):
        my_network.train_step(training_set,training_set_targets)

    
    
    
    
    
    
    
    