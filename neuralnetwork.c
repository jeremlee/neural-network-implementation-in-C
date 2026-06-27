#define _USE_MATH_DEFINES
#define GRADIENT_CLIP 5.0f


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
    neuralnetwork->epochCount = 0;
    neuralnetwork->learningRate = 0.0; //initial value of learning rate
    neuralnetwork->layers = (Layer*)malloc(neuralnetwork->layerCapacityCount*sizeof(Layer));
    neuralnetwork->layers[0].neuronCount = neuronCount;
    neuralnetwork->layers[0].values = (float*)malloc(neuronCount*sizeof(float));
    neuralnetwork->layers[0].biases = NULL; //input layer does not have biases
    neuralnetwork->layers[0].delta = NULL; //not applicable to input layer also
    neuralnetwork->history = NULL;
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



Dataset* createDataset(size_t row, size_t col, size_t targetCol, float input[row][col], float target[row][targetCol]){
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->row = row;
    dataset->col = col;
    dataset->targetCol = targetCol;
    dataset->input = (float*)malloc(row*col*sizeof(float));
    dataset->target = (float*)malloc(row*targetCol*sizeof(float));
    for(size_t i=0;i<row;i++){
        for(size_t j=0;j<col;j++){
            dataset->input[i*col+j] = input[i][j];
        }
    }
    //
    for(size_t i=0;i<row;i++){
        for(size_t j=0;j<targetCol;j++){
            dataset->target[i*targetCol+j] = target[i][j];
        }
    }
    return dataset;
}


NeuralNetwork* train(NeuralNetwork* neuralNetwork, Dataset* dataset, size_t epochs, float learningRate){
    if(neuralNetwork->history){
        free(neuralNetwork->history);
    }
    neuralNetwork->history = (Epoch*)malloc(epochs*sizeof(Epoch));
    neuralNetwork->epochCount = epochs;
    size_t outputLayerSize = neuralNetwork->layers[neuralNetwork->layerCount-1].neuronCount;
    float* forwardPassResults;
    float sum = 0;
    float loss = 0;
    float epochLoss;
    float reluZeroDivCheck, tanhZeroDivCheck, result;
    float input[dataset->col];
    size_t rowCount, colCount;
    for(size_t epoch=0;epoch<epochs;epoch++){ 
        epochLoss = 0;
        for(size_t datasetRow=0;datasetRow<dataset->row;datasetRow++){ // for each row in the dataset
             //forward pass START OF DATASET ROW
             for (size_t j=0;j<dataset->col;j++){
                input[j] = dataset->input[datasetRow*dataset->col+j];
            }
            forwardPassResults = computeOutput(neuralNetwork, input); // returns actual output  SLICE ARRAY
            switch(neuralNetwork->lossFunction){ // calculate delta values of output layer
                    case MSE:
                        for(size_t i=0;i<outputLayerSize;i++){
                            loss = (forwardPassResults[i] - dataset->target[datasetRow*dataset->targetCol+i]);
                            epochLoss+=powf(loss,2);
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
                        break;
                    case BINARY_CROSS_ENTROPY:
                        for(size_t i=0;i<outputLayerSize;i++){
                            loss = (forwardPassResults[i] - dataset->target[datasetRow*dataset->targetCol+i]);
                            //loss_i = -[y log(a) + (1 - y) log(1 - a)]
                            float p = forwardPassResults[i];
                            p = p < 1e-7f ? 1e-7f : p;
                            p = p > 1.0f - 1e-7f ? 1.0f - 1e-7f : p;
                            epochLoss += -((dataset->target[datasetRow*dataset->targetCol+i] * logf(forwardPassResults[i] < 1e-7f ? 1e-7f : 
                                            (forwardPassResults[i] > 1.0f-1e-7f ? 1.0f-1e-7f : forwardPassResults[i]))) 
                                            + ((1.0f-dataset->target[datasetRow*dataset->targetCol+i]) * 
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
                                    loss = forwardPassResults[i] - dataset->target[datasetRow*dataset->targetCol+i];
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
                        break;
                    case CATEGORICAL_CROSS_ENTROPY:
                        for(size_t i=0;i<outputLayerSize;i++){
                            loss = (forwardPassResults[i] - dataset->target[datasetRow*dataset->targetCol+i]);
                            epochLoss++;
                            switch(neuralNetwork->layers[neuralNetwork->layerCount-1].activationFunction){
                                case RELU:
                                    // delta = (a - y) * (a > 0 ? 1 : 0)
                                    result = loss * (forwardPassResults[i] > 0);
                                    if (result > GRADIENT_CLIP) result = GRADIENT_CLIP;
                                    if (result < -GRADIENT_CLIP) result = -GRADIENT_CLIP;
                                    neuralNetwork->layers[neuralNetwork->layerCount-1].delta[i] = result;
                                    break;
                                case SIGMOID:
                                    // delta = (a - y) * a * (1 - a)
                                    result = loss * forwardPassResults[i] * (1-forwardPassResults[i]);
                                    if (result > GRADIENT_CLIP) result = GRADIENT_CLIP;
                                    if (result < -GRADIENT_CLIP) result = -GRADIENT_CLIP;
                                    neuralNetwork->layers[neuralNetwork->layerCount-1].delta[i] = result;
                                    break;
                                case TANH:
                                    // delta = (a - y) * (1 - a * a)
                                    result = loss * (1 - forwardPassResults[i] * forwardPassResults[i]);
                                    if (result > GRADIENT_CLIP) result = GRADIENT_CLIP;
                                    if (result < -GRADIENT_CLIP) result = -GRADIENT_CLIP;
                                    neuralNetwork->layers[neuralNetwork->layerCount-1].delta[i] = result;
                                    break;
                                case NONE:
                                    break;
                            }
                        }
                        break;
                    default:
                        break;
                }

            // DELTA loop for previous layers
            for(int i=(int)neuralNetwork->layerCount-2;i>0;i--){ // for each layer going backward (back propagation)
                rowCount = neuralNetwork->layers[i].neuronCount;
                colCount = neuralNetwork->layers[i+1].neuronCount;
                for(size_t j=0;j<neuralNetwork->layers[i].neuronCount;j++){ // for each neuron in the current layer
                    sum = 0;
                    for(size_t k=0;k<neuralNetwork->layers[i+1].neuronCount;k++){ // for each current neuron's connections to next layer
                        // calculate deltas of each neuron
                        // sum += (weight to next * delta of next)
                        sum+=(neuralNetwork->weightMatrices[i].weightMatrix[j * colCount + k] * neuralNetwork->layers[i+1].delta[k]); 
                    }
                    // apply activation derivative
                    switch(neuralNetwork->layers[i].activationFunction){ 
                        case RELU:
                            // activation derivative = a > 0
                            sum*= (neuralNetwork->layers[i].values[j] > 0);
                            break;
                        case SIGMOID:
                            // activation derivative = a * (1 - a)
                            sum*= (neuralNetwork->layers[i].values[j] * (1 - neuralNetwork->layers[i].values[j]));
                            break;
                        case TANH:
                            // activation derivative =  1 - a^2
                            sum*= (1 - (neuralNetwork->layers[i].values[j] * neuralNetwork->layers[i].values[j]));
                            break;
                        case NONE:
                            break;
                    }
                    // assignment of deltas to neurons
                    neuralNetwork->layers[i].delta[j] = sum;
                }
                // end of DELTA loop
            }
        

            // WEIGHT UPDATE LOOP
            for(int currentLayer=(int)neuralNetwork->layerCount-1;currentLayer>0;currentLayer--){ // for each layer
                rowCount = neuralNetwork->layers[currentLayer-1].neuronCount;
                colCount = neuralNetwork->layers[currentLayer].neuronCount;
                for(size_t i=0;i<rowCount;i++){
                    for(size_t j=0;j<colCount;j++){
                        neuralNetwork->weightMatrices[currentLayer-1].weightMatrix[i*colCount+j] -=
                        learningRate*neuralNetwork->layers[currentLayer-1].values[i]*neuralNetwork->layers[currentLayer].delta[j];
                    }
                }
            }


            // BIAS UPDATE LOOP (put in another loop for the sake of my own sanity)
            for(size_t i=1;i<neuralNetwork->layerCount;i++){ // for each layer
                for(size_t j=0;j<neuralNetwork->layers[i].neuronCount;j++){ // for each neuron in layer
                    neuralNetwork->layers[i].biases[j] -= learningRate * neuralNetwork->layers[i].delta[j];
                }
            }

        } // END OF DATASET ROW
       
        epochLoss/=(neuralNetwork->layers[neuralNetwork->layerCount-1].neuronCount*dataset->row);
        neuralNetwork->history[epoch].loss = epochLoss;
        printf("Epoch: %zu/%zu\tLoss: %.4f\n", epoch+1, epochs, epochLoss); 
    } // end of epoch


    return neuralNetwork;
}

void showHistory(NeuralNetwork* neuralNetwork){
    for(size_t i=0;i<neuralNetwork->epochCount;i++){
        printf("Epoch: %zu/%zu\tLoss: %.4f\n", i+1, neuralNetwork->epochCount, neuralNetwork->history[i].loss);
    }
    return;
}