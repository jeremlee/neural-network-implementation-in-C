#ifndef _BCE_HANDLER_H_
#define _BCE_HANDLER_H_

#include"enum_activation_function_def.h"
#include"neural_network.h"
#include"defines.h"

typedef struct {
    ENUM_ACTIVATION_FUNCTION activationFunction;
    void(*handle)(NeuralNetwork*, Dataset*, size_t*, float*, float*, size_t*);
} BCEHandler;


void handleBCERelu(NeuralNetwork* neuralNetwork, Dataset* dataset, size_t* datasetRow, float* forwardPassResults, float* loss, size_t* i);
void handleBCESigmoid(NeuralNetwork* neuralNetwork, Dataset* dataset, size_t* datasetRow, float* forwardPassResults, float* loss, size_t* i);
void handleBCETanh(NeuralNetwork* neuralNetwork, Dataset* dataset, size_t* datasetRow, float* forwardPassResults, float* loss, size_t* i);


#endif