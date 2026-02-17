#ifndef NN_H
#define NN_H

typedef struct{
    int input_size;
    int hidden_size;
    int output_size;

    double *W1;
    double *b1;

    double *W2;
    double *b2;
}NeuralNetwork;


//initialise the neural network
void init_network(NeuralNetwork *nn,
                    int input_size,
                    int hidden_size,
                    int output_size);
void free_network(NeuralNetwork *nn);


//forward pass
void forward(NeuralNetwork *nn,
             double *input,
             double *hidden_output,
             double *final_output,
            double *hidden_z);


//back propagation
void backward(NeuralNetwork *nn,
              double *input,
              double *hidden_output,
              double *final_output,
              double *target,
              double learning_rate,
            double *hidden_z);

//activation function
double relu(double x);
double relu_derivative(double x);
void softmax(double *input, int size);

//save/load model in a .bin file
void save_model(NeuralNetwork *nn, const char *filename);
void load_model(NeuralNetwork *nn, const char *filename);

#endif