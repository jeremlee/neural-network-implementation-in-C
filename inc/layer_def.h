#ifndef _LAYER_DEF_H_
#define _LAYER_DEF_H_

#include"enum_activation_function_def.h"

/**

Represents a layer within the neural network.

biases - value of biases of each neuron 
values - actual value of the neurons in the layer
delta - error signal of the neuron (to be used in backpropagation)
neuronCount - number of neurons in the layer
activationFunction - type of activation function used in the layer


**/
typedef struct {
    float* biases;
    float* values;
    float* delta;
    size_t neuronCount;
    ENUM_ACTIVATION_FUNCTION activationFunction;
} Layer;


#endif 