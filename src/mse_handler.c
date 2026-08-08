
#include"neural_network.h"
#include"defines.h"
#include"macro_def.h"
#include"mse_handler.h"


void handleMSERelu(NeuralNetwork* neuralNetwork, float* forwardPassResults, float* loss, size_t* i){
    //delta = (a - y) * (a > 0)
    float result;
    int idx = *i;

    result = (*loss) * (forwardPassResults[idx] > 0);
    clip(result);

    neuralNetwork->layers[neuralNetwork->layerCount-1].delta[idx] = result;

    return;
}

void handleMSESigmoid(NeuralNetwork* neuralNetwork, float* forwardPassResults, float* loss, size_t* i){
    //delta = (a - y) * a * (1 - a)
    float result;
    int idx = *i;

    result = (*loss) * forwardPassResults[idx] * (1 - forwardPassResults[idx]);
    clip(result);

    neuralNetwork->layers[neuralNetwork->layerCount-1].delta[idx] = result;

    return;
}

void handleMSETanh(NeuralNetwork* neuralNetwork, float* forwardPassResults, float* loss, size_t* i){
    //delta = (a - y) * (1 - a^2)
    float result;
    int idx = *i;

    result = (*loss) * (1 - forwardPassResults[idx] * forwardPassResults[idx]);
    clip(result);

    neuralNetwork->layers[neuralNetwork->layerCount-1].delta[idx] = result;

    return;
}