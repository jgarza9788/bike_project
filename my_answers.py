"""
activate bike_sharing
"""

import numpy as np


class NeuralNetwork(object):

    def print_info(self):
        print(
            'input',self.input_nodes,'\n',
            'hidden_nodes',self.hidden_nodes,'\n',
            'output_nodes',self.output_nodes,'\n'
        )
        print(
            'w1',self.weights_input_to_hidden.shape,'\n'
            'w2',self.weights_hidden_to_output.shape,'\n'
        )

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                    (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                    (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        

        # print('self.weights_input_to_hidden.shape',self.weights_input_to_hidden.shape)
        # print('self.weights_hidden_to_output.shape',self.weights_hidden_to_output.shape)
        


        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #################################################################################
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        # self.activation_function = lambda x : self.sigmoid  # Replace 0 with your sigmoid calculation.
        
        self.activation_function = self.sigmoid
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    # def sigmoid_derivative(self,output):
    #     return output * (1 - output)

    # def sigmoid_prime(self,x):
    #     return self.sigmoid(x) * (1-self.sigmoid(x))

    # def error_formula(self,y, output):
    #     return - y*np.log(output) - (1 - y) * np.log(1-output)

    # def error_term_formula(self, x, y, output):
    #     return (y-output)*self.sigmoid(x)

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Implement forward pass here 

            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        #################################################################################
        # hidden_inputs = None # signals into hidden layer
        # hidden_outputs = None # signals from hidden layer

        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # TODO: Output layer - Replace these values with your calculations.
        #################################################################################
        # final_inputs = None # signals into final output layer
        # final_outputs = None # signals from final output layer

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        #final_outputs = self.sigmoid(final_inputs)
        final_outputs = final_inputs  # i don't need to sigmoid this
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # using this for another explainer on BP
        # https://www.youtube.com/watch?v=Ilg3gGewQ5U
        # https://www.youtube.com/watch?v=tIeHLnjs5U8

        # print('***backpropagation***')
        # print('final_outputs',final_outputs,final_outputs.shape)
        # print('hidden_outputs',hidden_outputs,hidden_outputs.shape)
        # print('X',X,X.shape)
        # print('y',y,y.shape)
        # print('delta_weights_i_h',delta_weights_i_h,delta_weights_i_h.shape)
        # print('delta_weights_h_o',delta_weights_h_o,delta_weights_h_o.shape)
        # print('******')

        # TODO: Output error - Replace this value with your calculations.
        #################################################################################
        # error = None # Output layer error is the difference between desired target and actual output.
        error =  y - final_outputs
        

        hidden_error = self.weights_hidden_to_output.dot(error)

        # TODO: Backpropagated error terms - Replace these values with your calculations.
        #################################################################################
        # output_error_term = None
        # output_error_term = error * self.sigmoid_derivative(final_outputs)
        # i guess if the output is one node we don't need the sigmoid derivative?
        output_error_term = error 


        # TODO: Calculate the hidden layer's contribution to the error
        #################################################################################

        # hidden_error_term = hidden_error * self.sigmoid_derivative(hidden_outputs)
        hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)
        
        # # Weight step (input to hidden)
        # delta_weights_i_h += None
        # # Weight step (hidden to output)
        # delta_weights_h_o += None

        # Weight step (input to hidden)
        # delta_weights_i_h += np.dot(X.reshape(3,1),hidden_error_term.reshape(1,2)) * self.lr
        #^doesn't always work (unable to reshape some data)
        delta_weights_i_h += hidden_error_term * X[:,None] #* self.lr


        # Weight step (hidden to output)
        #delta_weights_h_o +=  np.dot( hidden_outputs.reshape(2,1) , output_error_term.reshape(1,1) ) * self.lr
        #^doesn't always work (unable to reshape some data)
        delta_weights_h_o +=  output_error_term * hidden_outputs[:,None] #* self.lr


        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        # self.weights_hidden_to_output += None # update hidden-to-output weights with gradient descent step
        # self.weights_input_to_hidden += None # update input-to-hidden weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o/n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h/n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        #################################################################################
        # hidden_inputs = None # signals into hidden layer
        # hidden_outputs = None # signals from hidden layer
        # hidden_inputs = None # signals into hidden layer
        # hidden_outputs = None # signals from hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        #################################################################################
        # final_inputs = None # signals into final output layer
        # final_outputs = None # signals from final output layer 
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        # final_outputs = self.sigmoid(final_inputs)
        final_outputs = final_inputs
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 7500 
learning_rate = 0.75
hidden_nodes = 20
# hidden_nodes = 24
output_nodes = 1

#Training loss should be less than 0.09
#Validation loss should be less than 0.18

# Progress: 100.0% ...  Training loss: 0.063 ...  Validation loss: 0.144 ...  iterations 4999.0 out of 5000.0

# if __name__ == "__main__":
#     mnn = NeuralNetwork(3, 2, 1, 0.5)