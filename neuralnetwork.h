#include<stddef.h>

typedef enum{
    RELU,
    SIGMOID,
    TANH,
    NONE
} ActivationFunction;

typedef enum{
    MSE,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY
} LossFunction;

typedef struct Layer{
    float* biases;
    float* values;
    float* delta;
    size_t neuronCount;
    ActivationFunction activationFunction;
} Layer;

typedef struct WeightMatrix{
    size_t rowCount;
    size_t colCount;
    float* weightMatrix;
} WeightMatrix;

typedef struct NeuralNetwork{
    Layer* layers;
    WeightMatrix* weightMatrices;
    size_t layerCount;
    size_t layerCapacityCount;
    LossFunction lossFunction;
} NeuralNetwork;

NeuralNetwork* initialize(size_t neuronCount, LossFunction lossFunction);
NeuralNetwork* addLayer(NeuralNetwork* neuralNetwork, size_t neuronCount, ActivationFunction activationFunction);
float* computeOutput(NeuralNetwork* neuralNetwork, float* input);
void checkNetworkStatus(NeuralNetwork* neuralNetwork);