import math
import random


class neuron:
    def __init__(self,weights, input, activation_value):
        self.weights = weights
        self.activation_value = activation_value
        self.learning_rate = 1
        self.input = input
    def sigmoid(self,input):
        return 1/(1 + math.exp(-1*input))
    
    def activation_fn(self, input):
        self.activation_value = self.sigmoid(input)

    def weighted_sum(self):
        self.activation_fn(sum(val * weight for val,weight in zip(self.input , self.weights) ))
    
    # def update_weights(self):
    #     w_new = [self.weights - (self.learning_rate * self.weighted_sum(self.input))]
    #     self.weights = w_new

    def weight_mult(self):
        return [ val * weight  for val,weight in  zip(self.activation_value,self.weights)]


def main():
    input_neuron = neuron([random.randrange(-1,1,1)],[random.randrange(-1,1,1)],0.6)
    hidden_neuron = neuron([random.randrange(-1,1,1)],[input_neuron.activation_value],0.6)
    output_neuron = neuron([random.randrange(-1,1,1)],[hidden_neuron.activation_value],0.6)

    input_neuron.weighted_sum()
    hidden_neuron.weighted_sum()
    output_neuron.weighted_sum()


    print(output_neuron.activation_value)

main()