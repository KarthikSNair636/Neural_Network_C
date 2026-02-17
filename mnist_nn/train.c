#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"
#include "mnist_loader.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define EPOCHS 5
#define LEARNING_RATE 0.1

int main()
{
    srand(time(NULL));

    double *images;
    unsigned char *labels;
    int num_images, rows, cols;

    //Load MNIST training data
    load_mnist_images("data/train-images.idx3-ubyte",
                      &images,
                      &num_images,
                      &rows,
                      &cols);

    load_mnist_labels("data/train-labels.idx1-ubyte",
                      &labels,
                      &num_images);

    printf("Loaded %d training images\n", num_images);

    //Initialize network
    NeuralNetwork nn;
    init_network(&nn, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    double hidden_output[HIDDEN_SIZE];
    double final_output[OUTPUT_SIZE];
    double hidden_z[HIDDEN_SIZE];


    //Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        printf("Epoch %d/%d\n", epoch + 1, EPOCHS);

        for (int i = 0; i < num_images; i++)
        {
            double *input = &images[i * INPUT_SIZE];

            // Create target vector (one-hot)
            double target[OUTPUT_SIZE] = {0};
            target[labels[i]] = 1.0;

            // Forward pass
            forward(&nn, input, hidden_output, final_output,hidden_z);

            // Backpropagation
            backward(&nn,
                     input,
                     hidden_output,
                     final_output,
                     target,
                     LEARNING_RATE,
                    hidden_z);
        }
    }

    //Save trained model
    save_model(&nn, "model.bin");
    printf("Model saved to model.bin\n");

    //Free memory
    free_network(&nn);
    free(images);
    free(labels);

    return 0;
}


//compile using:gcc train.c nn.c mnist_loader.c -o train -lm
