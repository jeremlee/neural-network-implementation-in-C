#ifndef _LOSS_HANDLER_H_
#define _LOSS_HANDLER_H_

#include"enum_loss_function_def.h"
#include"neural_network.h"
#include"defines.h"

typedef struct {
    ENUM_LOSS_FUNCTION lossFunction;
    void(*handle)(NeuralNetwork*, size_t*, float*, Dataset*, float*);
} LossHandler;

void handleTrainingLossMSE(NeuralNetwork* neuralNetwork, size_t* datasetRow, float* epochLoss, Dataset* dataset, float* forwardPassResults);
void handleTrainingLossBCE(NeuralNetwork* neuralNetwork, size_t* datasetRow, float* epochLoss, Dataset* dataset, float* forwardPassResults);

#endif