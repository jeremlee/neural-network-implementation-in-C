#include<stddef.h>
#include<math.h>

#include"loss_handler.h"

void handleTrainingLossMSE(NeuralNetwork* neuralNetwork, size_t* datasetRow, float* epochLoss, Dataset* dataset, float* forwardPassResults){

    size_t outputLayerSize = neuralNetwork->layers[neuralNetwork->layerCount-1].neuronCount;
    float loss = 0;
    float result;


    for(size_t i=0;i<outputLayerSize;i++){
        loss = (forwardPassResults[i] - dataset->target[(*datasetRow)*dataset->targetCol+i]);
        (*epochLoss)+=powf(loss,2);
        switch(neuralNetwork->layers[neuralNetwork->layerCount-1].activationFunction){
            case RELU:
                //delta = (a - y) * (a > 0)

                result = loss * (forwardPassResults[i] > 0);
                if (result > GRADIENT_CLIP) result = GRADIENT_CLIP;
                if (result < -GRADIENT_CLIP) result = -GRADIENT_CLIP;
                neuralNetwork->layers[neuralNetwork->layerCount-1].delta[i] = result;

                break;
            case SIGMOID:
                //delta = (a - y) * a * (1 - a)
                result = loss * forwardPassResults[i] * (1 - forwardPassResults[i]);
                if (result > GRADIENT_CLIP) result = GRADIENT_CLIP;
                if (result < -GRADIENT_CLIP) result = -GRADIENT_CLIP;
                neuralNetwork->layers[neuralNetwork->layerCount-1].delta[i] = result;
                break;
            case TANH:
                //delta = (a - y) * (1 - a^2)
                result = loss * (1 - forwardPassResults[i] * forwardPassResults[i]);
                if (result > GRADIENT_CLIP) result = GRADIENT_CLIP;
                if (result < -GRADIENT_CLIP) result = -GRADIENT_CLIP;
                neuralNetwork->layers[neuralNetwork->layerCount-1].delta[i] = result;
                break;
            case NONE:
                break;
        }
    }
    return;
}

void handleTrainingLossBCE(NeuralNetwork* neuralNetwork, size_t* datasetRow, float* epochLoss, Dataset* dataset, float* forwardPassResults){

    size_t outputLayerSize = neuralNetwork->layers[neuralNetwork->layerCount-1].neuronCount;
    float loss = 0;
    float result;
    float tanhZeroDivCheck, reluZeroDivCheck; // avoid division by zero

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
        switch(neuralNetwork->layers[neuralNetwork->layerCount-1].activationFunction){
            case RELU:
                /// delta = ((a - y) / (a > 0 ? 1 : 0))
                reluZeroDivCheck = forwardPassResults[i] > 0 ? 1 : 0.000001f;
                result = loss / reluZeroDivCheck;
                if (result > GRADIENT_CLIP) result = GRADIENT_CLIP;
                if (result < -GRADIENT_CLIP) result = -GRADIENT_CLIP;
                neuralNetwork->layers[neuralNetwork->layerCount-1].delta[i] = result;
                break;
            case SIGMOID:
                // delta = (a - y) 
                loss = forwardPassResults[i] - dataset->target[(*datasetRow)*dataset->targetCol+i];
                result = loss;
                if (result > GRADIENT_CLIP) result = GRADIENT_CLIP;
                if (result < -GRADIENT_CLIP) result = -GRADIENT_CLIP;
                neuralNetwork->layers[neuralNetwork->layerCount-1].delta[i] = result;
                break;
            case TANH:
                // delta = ((a - y) / (1 - a*a))
                tanhZeroDivCheck = (1-forwardPassResults[i]*forwardPassResults[i]) == 0 ? 0.000001f : (1-forwardPassResults[i]*forwardPassResults[i]);
                result = loss / tanhZeroDivCheck;
                if (result > GRADIENT_CLIP) result = GRADIENT_CLIP;
                if (result < -GRADIENT_CLIP) result = -GRADIENT_CLIP;
                neuralNetwork->layers[neuralNetwork->layerCount-1].delta[i] = result;
                break;
            case NONE:
                break;
        }
    }

    return;
}