#ifndef _EPOCH_DEF_H_
#define _EPOCH_DEF_H_




/**

Represents an epoch.

loss - loss value of the network in the epoch
accuracy - accuracy of the network in the epoch


**/
typedef struct {
    float loss;
    float accuracy;
} Epoch;


#endif