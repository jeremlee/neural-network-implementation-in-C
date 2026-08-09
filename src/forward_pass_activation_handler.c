#include<math.h>
#include"forward_pass_activation_handler.h"



float handleRelu(float* sum){
    return (*sum) < 0 ? 0 : (*sum);
}

float handleSigmoid(float* sum){
    return 1.0f / (1.0f + expf(-(*sum)));
}

float handleTanh(float* sum){
    return tanhf((*sum));
}
