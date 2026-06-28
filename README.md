# Neural Network Implementation using C

Python is most commonly used for AI. I wanted to have
a refresher for C so I built a lightweight neural network library using C.

This project was built from scratch to explore how neural networks actually work, including manual memory management, weight initialization, forward pass, activation functions, and eventually backpropagation and training.

## Features

### PHASE 1: FORWARD PASS (COMPLETE)

* Fully connected neural networks
* Dynamic layer creation
* ReLU activation
* Sigmoid activation
* Tanh activation
* Random weight initialization
* Random bias initialization
* Forward pass/propagation

### PHASE 2: BACK PROPAGATION (MOSTLY IMPLEMENTED)

* Backpropagation and gradient descent
* Model serialization
* More activation and loss functions

### ISSUES:

* Cross-categorical entropy is not finished yet
* Special case handling not implemented yet (Like when only 2 layers are present (input, output only) in total)


## Example

```c
NeuralNetwork* nn = initialize(4, BINARY_CROSS_ENTROPY); //input layer with 4 neurons

nn = addLayer(nn, 8, RELU); //hidden layer with 8 neurons
nn = addLayer(nn, 2, SIGMOID); //output layer with 2 neurons

float input[] = {1, 0, 1, 0};

float* output = computeOutput(nn, input);
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
```

## Building

Using Clang:

```bash
clang main.c neuralnetwork.c -o main -lm
```

Using GCC:

```bash
gcc main.c neuralnetwork.c -o main -lm
```
Take note: the -lm is important because it uses the math library libm
Run:

```bash
./main
```
