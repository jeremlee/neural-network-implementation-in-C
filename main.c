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
    nn = addLayer(nn,2,SIGMOID);
    float input[] = {1,0,1,0};
    float* output = computeOutput(nn,input);
    for(size_t i=0;i<2;i++){
        printf("%.4f\n",output[i]);
    }
    return 0;
}