#include <stdio.h>
#include <stdlib.h>
#include "mnist_loader.h"

/*
    MNIST files are stored in big-endian format.
    Most computers are little-endian.
    So we must reverse the byte order.
*/
static int reverse_int(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) |
           ((int)c2 << 16) |
           ((int)c3 << 8)  |
           c4;
}

void load_mnist_images(const char *filename,
                       double **images,
                       int *num_images,
                       int *rows,
                       int *cols)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Error: Cannot open %s\n", filename);
        exit(1);
    }

    int magic_number = 0;

    fread(&magic_number, sizeof(int), 1, fp);
    magic_number = reverse_int(magic_number);

    if (magic_number != 2051)
    {
        printf("Invalid MNIST image file!\n");
        exit(1);
    }

    fread(num_images, sizeof(int), 1, fp);
    *num_images = reverse_int(*num_images);

    fread(rows, sizeof(int), 1, fp);
    *rows = reverse_int(*rows);

    fread(cols, sizeof(int), 1, fp);
    *cols = reverse_int(*cols);

    int image_size = (*rows) * (*cols);

    *images = malloc((*num_images) * image_size * sizeof(double));

    if (!*images)
    {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < (*num_images) * image_size; i++)
    {
        unsigned char pixel = 0;
        fread(&pixel, sizeof(unsigned char), 1, fp);

        // Normalize to 0–1
        (*images)[i] = pixel / 255.0;
    }

    fclose(fp);

    printf("Loaded %d images (%dx%d)\n",
           *num_images, *rows, *cols);
}

void load_mnist_labels(const char *filename,
                       unsigned char **labels,
                       int *num_labels)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Error: Cannot open %s\n", filename);
        exit(1);
    }

    int magic_number = 0;

    fread(&magic_number, sizeof(int), 1, fp);
    magic_number = reverse_int(magic_number);

    if (magic_number != 2049)
    {
        printf("Invalid MNIST label file!\n");
        exit(1);
    }

    fread(num_labels, sizeof(int), 1, fp);
    *num_labels = reverse_int(*num_labels);

    *labels = malloc((*num_labels) * sizeof(unsigned char));

    if (!*labels)
    {
        printf("Memory allocation failed\n");
        exit(1);
    }

    fread(*labels, sizeof(unsigned char), *num_labels, fp);

    fclose(fp);

    printf("Loaded %d labels\n", *num_labels);
}