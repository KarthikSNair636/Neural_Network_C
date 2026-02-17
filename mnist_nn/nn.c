#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include "nn.h"

//relu calculate the max(x,0)
//relu introduces non linearity
//used in forward pass
double relu(double x){
    if(x>0)
        return x;
    return 0;
}

//relu_derivative calculates the max(0,d(x)/dx)
//for gradient descent 
//used in back propagation
double relu_derivative(double x){
    if(x>0)
        return x;
    return 0;
}

//coverts the answers to probability
//used in output layer forward pass
void softmax(double *input,int size){
    double sum = 0;
    for(int i = 0;i<size;i++){
        input[i] = exp(input[i]);
        sum +=input[i];
    }
    for(int i = 0;i<size;i++){
        input[i] /= sum;
    }
}

//initialises the network
void init_network(NeuralNetwork *nn,
                    int input_size,
                    int hidden_size,
                    int output_size){

    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;

    nn->W1 = (double *)malloc(input_size*hidden_size*sizeof(double));
    nn->b1 = (double *)malloc(hidden_size*sizeof(double));

    nn->W2 = (double *)malloc(hidden_size*output_size*sizeof(double));
    nn->b2 = (double *)malloc(output_size*sizeof(double));

    for(int i = 0;i<input_size*hidden_size;i++)
        nn->W1[i] = (double)rand()/RAND_MAX-0.5;
    for(int i = 0;i<hidden_size*output_size;i++)
        nn->W2[i] = (double)rand()/RAND_MAX-0.5;

    for(int i = 0;i<hidden_size;i++)
        nn->b1[i] = 0;
    for(int i = 0;i<output_size;i++)
        nn->b2[i] = 0;
}

void free_network(NeuralNetwork *nn)
{
    free(nn->W1);
    free(nn->b1);
    free(nn->W2);
    free(nn->b2);
}

void forward(NeuralNetwork *nn,
             double *input,
             double *hidden_output,
             double *final_output,
            double *hidden_z){
    
    //hidden layer
    for(int j = 0;j<nn->hidden_size;j++){
        double sum = 0;
        for(int i = 0;i<nn->input_size;i++){
            sum +=input[i]*nn->W1[i*nn->hidden_size+j];
        }
        sum += nn->b1[j];
        hidden_z[j] = sum;
        hidden_output[j] = relu(sum);
    }

    //output layer
    for(int k = 0;k<nn->output_size;k++){
        double sum = 0;
        for(int j = 0;j<nn->hidden_size;j++){
            sum += hidden_output[j]*nn->W2[j*nn->output_size+k];
        }
        sum +=nn->b2[k];
        final_output[k]=sum;
    }
    softmax(final_output,nn->output_size);
}

void backward(NeuralNetwork *nn,
              double *input,
              double *hidden_output,
              double *final_output,
              double *target,
              double learning_rate,
            double *hidden_z)
{
    double output_error[nn->output_size];

    // Output error
    for (int i = 0; i < nn->output_size; i++)
        output_error[i] = final_output[i] - target[i];

    // Update W2 and b2
    for (int j = 0; j < nn->hidden_size; j++)
    {
        for (int k = 0; k < nn->output_size; k++)
        {
            nn->W2[j * nn->output_size + k] -=
                learning_rate *
                hidden_output[j] *
                output_error[k];
        }
    }

    for (int k = 0; k < nn->output_size; k++)
        nn->b2[k] -= learning_rate * output_error[k];

    // Hidden error
    double hidden_error[nn->hidden_size];

    for (int j = 0; j < nn->hidden_size; j++)
    {
        double sum = 0;

        for (int k = 0; k < nn->output_size; k++)
        {
            sum += output_error[k] *
                   nn->W2[j * nn->output_size + k];
        }

        hidden_error[j] =
            sum * relu_derivative(hidden_z[j]);
    }

    // Update W1 and b1
    for (int i = 0; i < nn->input_size; i++)
    {
        for (int j = 0; j < nn->hidden_size; j++)
        {
            nn->W1[i * nn->hidden_size + j] -=
                learning_rate *
                input[i] *
                hidden_error[j];
        }
    }

    for (int j = 0; j < nn->hidden_size; j++)
        nn->b1[j] -= learning_rate * hidden_error[j];
}

void save_model(NeuralNetwork *nn, const char *filename)
{
    FILE *fp = fopen(filename, "wb");

    fwrite(&nn->input_size, sizeof(int), 1, fp);
    fwrite(&nn->hidden_size, sizeof(int), 1, fp);
    fwrite(&nn->output_size, sizeof(int), 1, fp);

    fwrite(nn->W1, sizeof(double),
           nn->input_size * nn->hidden_size, fp);

    fwrite(nn->b1, sizeof(double),
           nn->hidden_size, fp);

    fwrite(nn->W2, sizeof(double),
           nn->hidden_size * nn->output_size, fp);

    fwrite(nn->b2, sizeof(double),
           nn->output_size, fp);

    fclose(fp);
}

void load_model(NeuralNetwork *nn, const char *filename)
{
    FILE *fp = fopen(filename, "rb");

    fread(&nn->input_size, sizeof(int), 1, fp);
    fread(&nn->hidden_size, sizeof(int), 1, fp);
    fread(&nn->output_size, sizeof(int), 1, fp);

    nn->W1 = malloc(nn->input_size * nn->hidden_size * sizeof(double));
    nn->b1 = malloc(nn->hidden_size * sizeof(double));
    nn->W2 = malloc(nn->hidden_size * nn->output_size * sizeof(double));
    nn->b2 = malloc(nn->output_size * sizeof(double));

    fread(nn->W1, sizeof(double),
          nn->input_size * nn->hidden_size, fp);

    fread(nn->b1, sizeof(double),
          nn->hidden_size, fp);

    fread(nn->W2, sizeof(double),
          nn->hidden_size * nn->output_size, fp);

    fread(nn->b2, sizeof(double),
          nn->output_size, fp);

    fclose(fp);
}
