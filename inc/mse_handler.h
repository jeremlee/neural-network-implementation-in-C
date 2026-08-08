#ifndef _MSE_HANDLER_H_
#define _MSE_HANDLER_H_

#include"enum_activation_function_def.h"
#include"neural_network.h"
#include"defines.h"

typedef struct {
    ENUM_ACTIVATION_FUNCTION activationFunction;
    void(*handle)(NeuralNetwork*, float*, float*, size_t*);
} MSEHandler;

void handleMSERelu(NeuralNetwork* neuralNetwork, float* forwardPassResults, float* loss, size_t* i);
void handleMSESigmoid(NeuralNetwork* neuralNetwork, float* forwardPassResults, float* loss, size_t* i);
void handleMSETanh(NeuralNetwork* neuralNetwork, float* forwardPassResults, float* loss, size_t* i);

#endif