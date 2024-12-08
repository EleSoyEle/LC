#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* readTextFile(char filename[]){
    FILE* file = fopen(filename,"r");
    if(file == NULL){
        perror("Error al abrir el archivo");
    }
    fseek(file,0,SEEK_END);
    long file_size = ftell(file);
    fseek(file,0,SEEK_SET);
    char* KernelS = (char*)malloc((file_size+1)*sizeof(char));
    fread(KernelS,1,file_size,file);
    fclose(file);
    KernelS[file_size]='\0';
    return KernelS;
}

void show_array(float* array,int size){
    printf("\n");
    for(int i=0;i<size;i++){
        printf("%f ",array[i]);
    }
    printf("\n");
}

void show_matrix(float** mat,int size[]){
    for(int i=0;i<size[0];i++){
        for(int j=0;j<size[1];j++){
            printf("%f ",mat[i][j]);
        }
        printf("\n");
    }
}

float* twod2oned(float** array,int s[2]){
    float* new_array = (float*)calloc(s[0]*s[1],sizeof(float));
    for(int i=0;i<s[0];i++){
        for(int j=0;j<s[1];j++){
            new_array[i*s[1]+j]=array[i][j];
        }
    }
    return new_array;
}

//Convierte una lista a una matriz de tamaño [size] a [size/dim,dim]
float** oned2twod(float* array,int size,int s[]){
    int num_filas = size/s[1];
    int num_col = size/s[0];
    float** new_array = (float**)calloc(num_filas,sizeof(float*));

    for(int i=0;i<num_filas;i++){
        new_array[i] = (float*)calloc(num_col,sizeof(float));
        for(int j=0;j<num_col;j++){
            new_array[i][j] = array[i*num_col+j];
        }
    }
    return new_array;
}