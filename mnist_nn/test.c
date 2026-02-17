#include <stdio.h>
#include <stdlib.h>
#include "nn.h"
#include "mnist_loader.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

int argmax(double *array, int size)
{
    int index = 0;
    double max = array[0];

    for (int i = 1; i < size; i++)
    {
        if (array[i] > max)
        {
            max = array[i];
            index = i;
        }
    }

    return index;
}

int main()
{
    double *images;
    unsigned char *labels;
    int num_images, rows, cols;

    // 1️⃣ Load trained model
    NeuralNetwork nn;
    load_model(&nn, "model.bin");

    printf("Model loaded successfully\n");

    // 2️⃣ Load test dataset
    load_mnist_images("data/t10k-images.idx3-ubyte",
                      &images,
                      &num_images,
                      &rows,
                      &cols);

    load_mnist_labels("data/t10k-labels.idx1-ubyte",
                      &labels,
                      &num_images);

    printf("Loaded %d test images\n", num_images);

    double hidden_output[HIDDEN_SIZE];
    double hidden_z[HIDDEN_SIZE];
    double final_output[OUTPUT_SIZE];

    int correct = 0;

    // 3️⃣ Testing loop
    for (int i = 0; i < num_images; i++)
    {
        double *input = &images[i * INPUT_SIZE];

        forward(&nn, input, hidden_output, final_output,hidden_z);

        int prediction = argmax(final_output, OUTPUT_SIZE);

        if (prediction == labels[i])
            correct++;
    }

    double accuracy = (double)correct / num_images * 100.0;

    printf("Accuracy: %.2f%%\n", accuracy);

    // 4️⃣ Free memory
    free_network(&nn);
    free(images);
    free(labels);

    return 0;
}
//compile as:gcc test.c nn.c mnist_loader.c -o test -lm
