#define _USE_MATH_DEFINES

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include "neuralnetwork.h"

NeuralNetwork* initialize(size_t neuronCount, LossFunction lossFunction){
    NeuralNetwork* neuralnetwork = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));   
    neuralnetwork->layerCount = 1;
    neuralnetwork->layerCapacityCount = 5;
    neuralnetwork->layers = (Layer*)malloc(neuralnetwork->layerCapacityCount*sizeof(Layer));
    neuralnetwork->layers[0].neuronCount = neuronCount;
    neuralnetwork->layers[0].values = (float*)malloc(neuronCount*sizeof(float));
    neuralnetwork->layers[0].biases = NULL; //input layer does not have biases
    neuralnetwork->layers[0].delta = NULL; //not applicable to input layer also
    neuralnetwork->layers[0].activationFunction = NONE; //not applicable also
    neuralnetwork->lossFunction = lossFunction;
    neuralnetwork->weightMatrices = (WeightMatrix*)malloc((neuralnetwork->layerCapacityCount-1)*sizeof(WeightMatrix));
    return neuralnetwork;
}

NeuralNetwork* addLayer(NeuralNetwork* neuralNetwork, size_t neuronCount, ActivationFunction activationFunction){
    if(!neuralNetwork){
        printf("Neural network is uninitialized!");
        exit(EXIT_FAILURE);
    }
    if(++(neuralNetwork->layerCount) == neuralNetwork->layerCapacityCount){
        //grow
        neuralNetwork->layerCapacityCount = (int)(neuralNetwork->layerCapacityCount * 1.25);
        neuralNetwork->layers = (Layer*)realloc(neuralNetwork->layers, neuralNetwork->layerCapacityCount*sizeof(Layer));
        neuralNetwork->weightMatrices = (WeightMatrix*)realloc(neuralNetwork->weightMatrices, neuralNetwork->layerCapacityCount*sizeof(WeightMatrix));
    }
    size_t currentLayer_index = neuralNetwork->layerCount-1;
    size_t currentWeightMatrix_index = currentLayer_index-1;
    Layer* currentLayer = &neuralNetwork->layers[currentLayer_index];
    Layer* previousLayer = &neuralNetwork->layers[currentLayer_index-1];
    WeightMatrix* currentWeightMatrix = &neuralNetwork->weightMatrices[currentWeightMatrix_index];
    currentLayer->neuronCount = neuronCount;
    currentLayer->values = (float*)malloc(neuronCount*sizeof(float));
    currentLayer->biases = (float*)malloc(neuronCount*sizeof(float));
    float bias_min_val = -0.001;
    float bias_max_val = 0.001;
    for(size_t i=0;i<currentLayer->neuronCount;i++){
        currentLayer->biases[i] = bias_min_val + ((float)rand()/(float)RAND_MAX) * (bias_max_val-bias_min_val);
    }
    currentLayer->delta = (float*)malloc(neuronCount*sizeof(float));
    currentLayer->activationFunction = activationFunction;
    currentWeightMatrix->rowCount = previousLayer->neuronCount;
    currentWeightMatrix->colCount = currentLayer->neuronCount;
    currentWeightMatrix->weightMatrix = (float*)malloc(currentWeightMatrix->rowCount*currentWeightMatrix->colCount*sizeof(float));
    float weight_min_val = (float)((-1)/sqrtf(currentWeightMatrix->rowCount));
    float weight_max_val = (float)(1/sqrtf(currentWeightMatrix->rowCount));
    for(size_t row=0;row<currentWeightMatrix->rowCount;row++){
        for(size_t col=0;col<currentWeightMatrix->colCount;col++){
            currentWeightMatrix->weightMatrix[row*currentWeightMatrix->colCount+col] = weight_min_val + ((float)rand()/(float)RAND_MAX) * (weight_max_val-weight_min_val);
        }
    }
    return neuralNetwork;
}

float* computeOutput(NeuralNetwork* neuralNetwork, float* input){
    Layer inputLayer = neuralNetwork->layers[0];
    Layer* previousLayer;
    Layer* currentLayer;
    WeightMatrix* currentMatrix;
    size_t rowCount, colCount;
    float sum = 0.0;
    memcpy(neuralNetwork->layers[0].values,input,inputLayer.neuronCount*sizeof(float));
    for(size_t i=1;i<neuralNetwork->layerCount;i++){
        previousLayer = &neuralNetwork->layers[i-1];
        currentLayer = &neuralNetwork->layers[i];
        currentMatrix = &neuralNetwork->weightMatrices[i-1];
        rowCount = previousLayer->neuronCount;
        colCount = currentLayer->neuronCount;
        for(size_t current=0;current<colCount;current++){
            sum = currentLayer->biases[current];
            for(size_t previous=0;previous<rowCount;previous++){ //calculation
                sum+=(previousLayer->values[previous] * currentMatrix->weightMatrix[previous*colCount+current]);
            }
            //activation
            float finalNeuronValue;
            switch(currentLayer->activationFunction){
                case RELU:
                    finalNeuronValue = sum < 0 ? 0 : sum;
                    break;
                case SIGMOID:
                    finalNeuronValue = 1.0f / (1.0f + expf(-sum));
                    break;
                case TANH:
                    finalNeuronValue = tanhf(sum);
                    break;
                default:
                    finalNeuronValue = sum;
                    break;
            }
            currentLayer->values[current] = finalNeuronValue;
        }
    }
    return neuralNetwork->layers[neuralNetwork->layerCount-1].values;
}

void checkNetworkStatus(NeuralNetwork* neuralNetwork){
    printf("NN STATUS:\n");
    printf("Layer Count: %zu\n", neuralNetwork->layerCount);
    printf("Layer Capacity Count: %zu\n", neuralNetwork->layerCapacityCount);
    return;
}