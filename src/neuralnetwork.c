#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include"neural_network.h"
#include"defines.h"
#include"loss_handler.h"
#include"enum_activation_function_def.h"
#include"enum_loss_function_def.h"
#include"forward_pass_activation_handler.h"

const ForwardPassActivationHandler fpActivationHandler[] = {
    {RELU, &handleRelu},
    {SIGMOID, &handleSigmoid},
    {TANH, &handleTanh}
};

size_t fpActivationHandlerSize = sizeof(fpActivationHandler) / sizeof(fpActivationHandler[0]);

const LossHandler lossHandler[] = {
    {MSE, &handleTrainingLossMSE},
    {BINARY_CROSS_ENTROPY, &handleTrainingLossBCE}
};

size_t lossHandlerSize = sizeof(lossHandler) / sizeof(lossHandler[0]);


NeuralNetwork* initialize(size_t neuronCount, ENUM_LOSS_FUNCTION lossFunction){
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

NeuralNetwork* addLayer(NeuralNetwork* neuralNetwork, size_t neuronCount, ENUM_ACTIVATION_FUNCTION activationFunction){
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

    for(size_t i=0;i<currentLayer->neuronCount;i++){
        currentLayer->biases[i] = BIAS_MIN_VAL + ((float)rand()/(float)RAND_MAX) * (BIAS_MAX_VAL-BIAS_MIN_VAL);
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
            currentWeightMatrix->weightMatrix[row*currentWeightMatrix->colCount+col] = weight_min_val + ((float)rand()/(float)RAND_MAX) * 
                                                                                        (weight_max_val-weight_min_val);
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
            float finalNeuronValue = 0.0f;

            for(size_t j=0;j<fpActivationHandlerSize;j++){
                if(currentLayer->activationFunction == fpActivationHandler[j].activationFunction){
                    finalNeuronValue = fpActivationHandler[j].handle(&sum);
                }
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
    
    float* forwardPassResults;
    float sum = 0;
    float epochLoss;
    float input[dataset->col];
    size_t rowCount, colCount;

    for(size_t epoch=0;epoch<epochs;epoch++){ 
        epochLoss = 0;
        for(size_t datasetRow=0;datasetRow<dataset->row;datasetRow++){ // for each row in the dataset
             //forward pass START OF DATASET ROW
            for (size_t j=0;j<dataset->col;j++){
                input[j] = dataset->input[datasetRow*dataset->col+j];
            }
            forwardPassResults = computeOutput(neuralNetwork, input); // returns actual output 
            
            for(size_t i=0;i<lossHandlerSize;i++){
                if(neuralNetwork->lossFunction == lossHandler[i].lossFunction){
                    lossHandler[i].handle(neuralNetwork, &datasetRow, &epochLoss, dataset, forwardPassResults);
                }
            }
            //end forward pass


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