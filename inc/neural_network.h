#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_


#include<stddef.h>
#include"enum_loss_function_def.h"
#include"enum_activation_function_def.h"
#include"layer_def.h"
#include"epoch_def.h"
#include"weight_matrix_def.h"
#include"dataset_def.h"


/**

Represents the whole neural network.

layers - array containing the layers of the network.
weightMatrixes - array containing the weight matrices between layers
layerCount - number of actual layers in the network
layerCapacityCount - number of layers allocated in the network (will grow as needed)
learningRate - learning rate of the neural network.
lossFunction - type of loss function used in the network
history - array of epochs containing data like loss and accuracy during the training

**/
typedef struct NeuralNetwork{
    Layer* layers;
    WeightMatrix* weightMatrices;
    size_t layerCount;
    size_t layerCapacityCount;
    size_t epochCount;
    float learningRate;
    ENUM_LOSS_FUNCTION lossFunction;
    Epoch* history;
} NeuralNetwork;

/**
 
Creates a new neural network and initializes the input layer.

@param neuronCount Number of neurons in the input layer.
@param lossFunction Loss function used during training.
@return Pointer of type NeuralNetwork to the newly allocated neural network.
 
**/
NeuralNetwork* initialize(size_t neuronCount, ENUM_LOSS_FUNCTION lossFunction);

/**

Adds a fully connected layer to the neural network.

Allocates neuron values, biases, deltas, and the weight matrix
connecting the previous layer to the new layer.

@param neuralNetwork Neural network instance.
@param neuronCount Number of neurons in the new layer.
@param activationFunction Activation function for the layer.
@return Pointer to the neural network.

**/
NeuralNetwork* addLayer(NeuralNetwork* neuralNetwork, size_t neuronCount, ENUM_ACTIVATION_FUNCTION activationFunction);


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

@return nothing

**/
void checkNetworkStatus(NeuralNetwork* neuralNetwork);



/**

Trains the neural network on the target output data.

@return Pointer to the neural network.

**/
NeuralNetwork* train(NeuralNetwork* neuralNetwork, Dataset* dataset, size_t epochs, float learningRate);



/**

Shows the history of the neural network's training

@return nothing

**/
void showHistory(NeuralNetwork* neuralNetwork);



/**

Creates a dataset

@param row Number of rows in the matrix
@param col Number of columns in the matrix
@param input The inputs of the dataset
@param target The desired outputs
@return Newly created dataset pointer

**/
Dataset* createDataset(size_t row, size_t col, size_t targetCol, float input[row][col], float target[row][targetCol]);


void handleTrainingLossMSE(NeuralNetwork* neuralNetwork, size_t* datasetRow, float* epochLoss, Dataset* dataset, float* forwardPassResults);
void handleTrainingLossBCE(NeuralNetwork* neuralNetwork, size_t* datasetRow, float* epochLoss, Dataset* dataset, float* forwardPassResults);


#endif