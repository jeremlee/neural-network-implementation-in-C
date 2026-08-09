#ifndef _FORWARD_PASS_ACTIVATION_HANDLER_H_
#define _FORWARD_PASS_ACTIVATION_HANDLER_H_

#include"enum_activation_function_def.h"
#include"neural_network.h"
#include"defines.h"

typedef struct {
    ENUM_ACTIVATION_FUNCTION activationFunction;
    float(*handle)(float*);
} ForwardPassActivationHandler;

float handleRelu(float* sum);
float handleSigmoid(float* sum);
float handleTanh(float* sum);

#endif