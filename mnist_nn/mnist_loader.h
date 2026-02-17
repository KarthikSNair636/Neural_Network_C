#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

// Load images from .idx3-ubyte file
void load_mnist_images(const char *filename,
                       double **images,
                       int *num_images,
                       int *rows,
                       int *cols);

// Load labels from .idx1-ubyte file
void load_mnist_labels(const char *filename,
                       unsigned char **labels,
                       int *num_labels);

#endif
