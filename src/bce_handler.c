#include"neural_network.h"
#include"defines.h"
#include"macro_def.h"
#include"bce_handler.h"


void handleBCERelu(NeuralNetwork* neuralNetwork, Dataset* dataset, size_t* datasetRow, float* forwardPassResults, float* loss, size_t* i){
    /// delta = ((a - y) / (a > 0 ? 1 : 0))
    size_t idx = *i;
    float reluZeroDivCheck = forwardPassResults[idx] > 0 ? 1 : 0.000001f;

    float result = (*loss) / reluZeroDivCheck;

    clip(result);
    neuralNetwork->layers[neuralNetwork->layerCount-1].delta[idx] = result;

    return;
}

void handleBCESigmoid(NeuralNetwork* neuralNetwork, Dataset* dataset, size_t* datasetRow, float* forwardPassResults, float* loss, size_t* i){
    // delta = (a - y)     
    size_t idx = *i;
    float reluZeroDivCheck = forwardPassResults[idx] > 0 ? 1 : 0.000001f;
    *loss = forwardPassResults[idx] - dataset->target[(*datasetRow)*dataset->targetCol+idx];

    
    float result = (*loss) / reluZeroDivCheck;

    clip(result);
    neuralNetwork->layers[neuralNetwork->layerCount-1].delta[idx] = result;

    return;
}

void handleBCETanh(NeuralNetwork* neuralNetwork, Dataset* dataset, size_t* datasetRow, float* forwardPassResults, float* loss, size_t* i){
    // delta = ((a - y) / (1 - a*a))
    size_t idx = *i;
    float tanhZeroDivCheck = (1-forwardPassResults[idx]*forwardPassResults[idx]) == 0 ? 0.000001f : (1-forwardPassResults[idx]*forwardPassResults[idx]);
    float result = (*loss) / tanhZeroDivCheck;

    clip(result);
    neuralNetwork->layers[neuralNetwork->layerCount-1].delta[idx] = result;

    return;
}
