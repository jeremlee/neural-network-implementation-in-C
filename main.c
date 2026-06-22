#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"neuralnetwork.h"
int main(){
    srand(time(NULL));
    printf("Forward pass test\n");
    NeuralNetwork* nn = initialize(4,BINARY_CROSS_ENTROPY);
    checkNetworkStatus(nn);
    for(size_t i=0;i<15;i++){
        nn = addLayer(nn,8,RELU);
        checkNetworkStatus(nn);
    }
    nn = addLayer(nn,1,SIGMOID);
    float input[] = {1,0,1,0};
    float* output = computeOutput(nn,input);
    for(size_t i=0;i<2;i++){
        printf("%.4f\n",output[i]);
    }
    float train_input[] = {1,1,1,1};
    float target_data[] = {1};

    nn = train(nn,train_input,target_data,1000,0.01f);
    //showHistory(nn);

    return 0;
}