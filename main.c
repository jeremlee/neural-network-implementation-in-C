#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"neuralnetwork.h"
int main(){
    srand(42);
    printf("Forward pass test\n");
    NeuralNetwork* nn = initialize(4,BINARY_CROSS_ENTROPY);
    nn = addLayer(nn,4,RELU);
    nn = addLayer(nn,1,SIGMOID);
    checkNetworkStatus(nn);
    float input[] = {1,0,1,0};
    float* output = computeOutput(nn,input);
    for(size_t i=0;i<2;i++){
        printf("%.4f\n",output[i]);
    }

    // simulate XOR
    float train_input[16][4] = {
        {1, 1, 1, 1},
        {1, 1, 1, 0},
        {1, 1, 0, 1},
        {1, 1, 0, 0},
        {1, 0, 1, 1},
        {1, 0, 1, 0},
        {1, 0, 0, 1},
        {1, 0, 0, 0},
        {0, 1, 1, 1},
        {0, 1, 1, 0},
        {0, 1, 0, 1},
        {0, 1, 0, 0},
        {0, 0, 1, 1},
        {0, 0, 1, 0},
        {0, 0, 0, 1},
        {0, 0, 0, 0}
    };
    float target_data[16][1] = {
        {0}, // {1, 1, 1, 1} -> 4 ones (even)
        {1}, // {1, 1, 1, 0} -> 3 ones (odd)
        {1}, // {1, 1, 0, 1} -> 3 ones (odd)
        {0}, // {1, 1, 0, 0} -> 2 ones (even)
        {1}, // {1, 0, 1, 1} -> 3 ones (odd)
        {0}, // {1, 0, 1, 0} -> 2 ones (even)
        {0}, // {1, 0, 0, 1} -> 2 ones (even)
        {1}, // {1, 0, 0, 0} -> 1 one  (odd)
        {1}, // {0, 1, 1, 1} -> 3 ones (odd)
        {0}, // {0, 1, 1, 0} -> 2 ones (even)
        {0}, // {0, 1, 0, 1} -> 2 ones (even)
        {1}, // {0, 1, 0, 0} -> 1 one  (odd)
        {0}, // {0, 0, 1, 1} -> 2 ones (even)
        {1}, // {0, 0, 1, 0} -> 1 one  (odd)
        {1}, // {0, 0, 0, 1} -> 1 one  (odd)
        {0}  // {0, 0, 0, 0} -> 0 ones (even)
    };
    Dataset* dataset = createDataset(16, 4, 1, train_input, target_data);
    nn = train(nn,dataset,676767,0.005f);
    //showHistory(nn);

    return 0;
}