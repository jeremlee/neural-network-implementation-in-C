#include<stddef.h>
#include<math.h>

#include"loss_handler.h"
#include"macro_def.h"
#include"mse_handler.h"
#include"bce_handler.h"

const MSEHandler mseHandler[] = {
    {RELU, &handleMSERelu},
    {SIGMOID, &handleMSESigmoid},
    {TANH, &handleMSETanh}
};

size_t mseHandlerSize = sizeof(mseHandler) / sizeof(mseHandler[0]);

const BCEHandler bceHandler[] = {
    {RELU, &handleBCERelu},
    {SIGMOID, &handleBCESigmoid},
    {TANH, &handleBCETanh}
};

size_t bceHandlerSize = sizeof(bceHandler) / sizeof(bceHandler[0]);




void handleTrainingLossMSE(NeuralNetwork* neuralNetwork, size_t* datasetRow, float* epochLoss, Dataset* dataset, float* forwardPassResults){

    size_t outputLayerSize = neuralNetwork->layers[neuralNetwork->layerCount-1].neuronCount;
    float loss = 0;
  
    for(size_t i=0;i<outputLayerSize;i++){
        loss = (forwardPassResults[i] - dataset->target[(*datasetRow)*dataset->targetCol+i]);
        (*epochLoss)+=powf(loss,2);

        for(size_t j=0;j<mseHandlerSize;j++){
            if(neuralNetwork->layers[neuralNetwork->layerCount-1].activationFunction == mseHandler[j].activationFunction){
                mseHandler[j].handle(neuralNetwork, forwardPassResults, &loss, &i);
            }
        }

    }
    return;
}


void handleTrainingLossBCE(NeuralNetwork* neuralNetwork, size_t* datasetRow, float* epochLoss, Dataset* dataset, float* forwardPassResults){

    size_t outputLayerSize = neuralNetwork->layers[neuralNetwork->layerCount-1].neuronCount;
    float loss = 0;

    for(size_t i=0;i<outputLayerSize;i++){
        loss = (forwardPassResults[i] - dataset->target[(*datasetRow)*dataset->targetCol+i]);
        //loss_i = -[y log(a) + (1 - y) log(1 - a)]
        float p = forwardPassResults[i];
        p = p < 1e-7f ? 1e-7f : p;
        p = p > 1.0f - 1e-7f ? 1.0f - 1e-7f : p;
        (*epochLoss) += -((dataset->target[(*datasetRow)*dataset->targetCol+i] * logf(forwardPassResults[i] < 1e-7f ? 1e-7f : 
                        (forwardPassResults[i] > 1.0f-1e-7f ? 1.0f-1e-7f : forwardPassResults[i]))) 
                        + ((1.0f-dataset->target[(*datasetRow)*dataset->targetCol+i]) * 
                        logf(1.0f - (forwardPassResults[i] < 1e-7f ? 1e-7f : 
                        (forwardPassResults[i] > 1.0f-1e-7f ? 1.0f-1e-7f : forwardPassResults[i])))));
        for(size_t j=0;j<bceHandlerSize;j++){
            if(bceHandler[j].activationFunction == neuralNetwork->layers[neuralNetwork->layerCount-1].activationFunction){
                bceHandler[j].handle(neuralNetwork, dataset, datasetRow, forwardPassResults, &loss, &i);
            }
        }
    }

    return;
}