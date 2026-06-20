#include<stddef.h>


/**

Activation functions supported currently
 
**/
typedef enum{
    RELU,
    SIGMOID,
    TANH,
    NONE
} ActivationFunction;


/**

Loss functions supported currently

MSE - Mean Squared Error (typically used for regression)
BINARY CROSS ENTROPY - used for binary classification
CATEGORICAL CROSS ENTROPY - used for multi-class classification
 
**/
typedef enum{
    MSE,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY
} LossFunction;


/**

Represents a layer within the neural network.

biases - value of biases of each neuron 
values - actual value of the neurons in the layer
delta - error signal of the neuron (to be used in backpropagation)
neuronCount - number of neurons in the layer
activationFunction - type of activation function used in the layer


**/
typedef struct Layer{
    float* biases;
    float* values;
    float* delta;
    size_t neuronCount;
    ActivationFunction activationFunction;
} Layer;


/**

Represents a weight matrix connecting a layer to another.

rowCount - number of neurons in the previous layer
colCount - number of neurons in the current layer
weightMatrix - actual matrix of weight values represented in a 1D matrix through indexing
 
**/
typedef struct WeightMatrix{
    size_t rowCount;
    size_t colCount;
    float* weightMatrix;
} WeightMatrix;


/**

Represents the whole neural network.

layers - array containing the layers of the network.
weightMatrixes - array containing the weight matrices between layers
layerCount - number of actual layers in the network
layerCapacityCount - number of layers allocated in the network (will grow as needed)
learningRate - learning rate of the neural network.
lossFunction - type of loss function used in the network
 
**/
typedef struct NeuralNetwork{
    Layer* layers;
    WeightMatrix* weightMatrices;
    size_t layerCount;
    size_t layerCapacityCount;
    float learningRate;
    LossFunction lossFunction;
} NeuralNetwork;

/**
 
Creates a new neural network and initializes the input layer.

@param neuronCount Number of neurons in the input layer.
@param lossFunction Loss function used during training.
@return Pointer of type NeuralNetwork to the newly allocated neural network.
 
**/
NeuralNetwork* initialize(size_t neuronCount, LossFunction lossFunction);

/**

Adds a fully connected layer to the neural network.

Allocates neuron values, biases, deltas, and the weight matrix
connecting the previous layer to the new layer.

@param neuralNetwork Neural network instance.
@param neuronCount Number of neurons in the new layer.
@param activationFunction Activation function for the layer.
@return Pointer to the neural network.

**/
NeuralNetwork* addLayer(NeuralNetwork* neuralNetwork, size_t neuronCount, ActivationFunction activationFunction);


/**

Performs a forward pass through the network.

Copies the input values into the input layer and computes
the output of every layer.

@param neuralNetwork Neural network instance.
@param input Input vector.
@return Pointer to the output layer values.

**/
float* computeOutput(NeuralNetwork* neuralNetwork, float* input);


/**

Checks the number of layers and number of layers allocated in the neural network.

@return Pointer to the output layer values.

**/
void checkNetworkStatus(NeuralNetwork* neuralNetwork);



/**

Trains the neural network on the target output data.

@return Pointer to the neural network.

**/
NeuralNetwork* train(NeuralNetwork* neuralNetwork, float* input, float* target, size_t epochs, float learningRate);
