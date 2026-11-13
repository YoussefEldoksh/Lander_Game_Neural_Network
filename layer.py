from neuron import neuron

class layer:
    def __init__(self, neurons: list[neuron] = None):
        self.neurons = neurons or []
        self.outputs = []
        self.errors = []
        self.learning_rate = 0.01
        self.Lambda = 0.8
        self.momentum = 0.1
        
    def feedforward(self,inputs):
        self.outputs = [curr_neuron.feedforward(inputs) for curr_neuron in self.neurons]
        return self.outputs
    
    def backpropagation(self, targets = None, prev_layer_outputs = None , next_layer = None):
        if targets is not None:
            self.errors = [target - prediction for target, prediction in zip(targets,self.outputs)]
            print(self.errors)
        else:
            self.errors = [
                sum(n.local_grad * n.old_weights[j] for n in next_layer.neurons)
                for j in range(len(self.neurons))
            ]
            print(self.errors)
            
        for neuron , error in zip(self.neurons , self.errors):
            neuron.compute_loc_grad(error)
        
        for neuron in self.neurons:
            neuron.update_weights(prev_layer_outputs)
        