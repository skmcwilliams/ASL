Function model takes the below function as parameters in order to
read hand signals showing numbers 1-5. The model applies AdamOptimization
with l2 regularization and learning rate of 0.0001. Model runs 1500
epochs and then prints the cost for every 100 epochs. The ultimate
accuracy of this model is 100% for training and 87% for test. Cost results plotted in graph.

linear function: First Layer of NN
sigmoid: Second Layer of NN
cost: compute performance
one_hot_matrix: convert to one-hot encoding
ones: create matrix of ones
create_placeholders: placeholders needed for the model
initialize_parameters: crate W and b parameters for model
forward_propagation: identify Z and A, return Z3 for final prediction
compute_cost: compute cost of Z3