#ifndef _WEIGHT_MATRIX_DEF_H_
#define _WEIGHT_MATRIX_DEF_H_



/**

Represents a weight matrix connecting a layer to another.

rowCount - number of neurons in the previous layer
colCount - number of neurons in the current layer
weightMatrix - actual matrix of weight values represented in a 1D matrix through indexing
 
**/
typedef struct WeightMatrix{
    size_t rowCount;
    size_t colCount;
    float* weightMatrix;
} WeightMatrix;

#endif