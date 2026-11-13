import random


class neuron:
    def __init__(self,weights):
        self.weights = weights
        self.old_weights = []
        self.activation_value = 0
        self.learning_rate = 0.01
        self.lamb = 0.8
        self.momentum = 0.1
        self.bias = 1
        self.prev_deltas = [0.0] * len(weights)
        self.prev_deltas_of_bias = 0.0
        self.local_grad = 0.0

    def sigmoid(self,input):
        return 1/(1 + 2.72**(-1*input*self.lamb))
    
    def activation_fn(self, input):
        self.activation_value = self.sigmoid(input)
        return self.activation_value

    def feedforward(self,input):
        return self.activation_fn(sum(val * weight for val,weight in zip(input , self.weights)) + self.bias)
    
    def compute_loc_grad(self, error):
        
        self.local_grad = self.lamb* self.activation_value * error * (1-self.activation_value)

    def update_weights(self, prev_layer_outputs):
        print(f"old weights are: {self.weights}")
        self.old_weights = self.weights
        for i,prev_layer_output in enumerate(prev_layer_outputs):
            delta_ij = self.learning_rate * self.local_grad * prev_layer_output + (self.momentum * self.prev_deltas[i])
            self.weights[i] += delta_ij
            self.prev_deltas[i] = delta_ij
            
        delta_ij = self.learning_rate * self.local_grad + (self.momentum * self.prev_deltas_of_bias)
        self.bias += delta_ij
        self.prev_deltas_of_bias = delta_ij
        print(f"new weights are: {self.weights}")
        
