#ifndef _DATASET_DEF_H_
#define _DATASET_DEF_H_

typedef struct Dataset{
    size_t row;
    size_t col;
    size_t targetCol;
    float* input;
    float* target;
} Dataset;


#endif