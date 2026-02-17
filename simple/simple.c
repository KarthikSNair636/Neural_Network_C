#include <stdio.h>
#include <stdlib.h>
#include<time.h>

#define INP 10
#define OUT 10

int main(){
    int i,j;

    srand(time(NULL));

    float *input = (float *)malloc(INP*sizeof(float));
    float *bias = (float *)malloc(OUT*sizeof(float));
    float *weights = (float *)malloc(OUT*INP*sizeof(float));
    float *layer_output = (float*)malloc(OUT*sizeof(float));

    if(!input || !bias || !weights || !layer_output){
        printf("Memory allocation failed\n");
        return 1;
    }


    for(i = 0;i<INP;i++){
        input[i] = (float)rand()/RAND_MAX*10;
    }

    for(i = 0;i<OUT;i++){
        bias[i]=(float)rand()/RAND_MAX*10;
        for(j = 0;j<INP;j++){
            weights[i*INP+j] = (float)rand()/RAND_MAX*10;
        }
    }

    for(i = 0;i<OUT;i++){
        layer_output[i] = 0;
        for(j = 0;j<INP;j++)
            layer_output[i] += input[j]*weights[i*INP+j];
        layer_output[i]+=bias[i];
    }

    printf("\n\n##### Inputs #####\n\n");
    for(i = 0;i<INP;i++)
        printf("%.2f\t",input[i]);

    printf("\n\n##### Biases #####\n\n");
    for(i = 0;i<OUT;i++)
        printf("%.2f\t",bias[i]);

    printf("\n\n##### Weights #####\n\n");
    for(i = 0;i<OUT;i++){
        for(j = 0;j<INP;j++)
            printf("%.2f\t",weights[i*INP+j]);
        printf("\n");
    }

    printf("\n\n##### Layer Output #####\n\n");
    for(i =0;i<OUT;i++)
        printf("%.2f\t",layer_output[i]);

    printf("\n");

    free(input);
    free(bias);
    free(weights);
    free(layer_output);

    return 0;
}