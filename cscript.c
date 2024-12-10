//Advertencia, este codigo es tan complejo porque esta optimizado para usar recursos del cpu y gpu en paralelo
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/opencl.h>
#include "utils.c"


cl_int qerror = CL_SUCCESS;
cl_int cerror = CL_SUCCESS;


cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;
cl_device_type devt;

int init_opencl(){
    const char* KernelSource = readTextFile("kernel.cl");
    const cl_uint num = 1;
    cl_device_type devt = CL_DEVICE_TYPE_CPU;
    clGetDeviceIDs(NULL,devt,0,NULL,(cl_uint*)&num);
    cl_device_id devices[1];
    clGetDeviceIDs(NULL,devt,num,devices,NULL);

    context = clCreateContextFromType(NULL,devt,NULL,NULL,&cerror);
    if(cerror != CL_SUCCESS){
        printf("Error en el context \n");
    }
    clGetDeviceIDs(NULL,devt,1,devices,NULL);

    queue = clCreateCommandQueueWithProperties(context,devices[0],NULL,&qerror);
    if(qerror != CL_SUCCESS){
        printf("Error en el comand queue \n");
    }
    program = clCreateProgramWithSource(context,1,(const char**)&KernelSource,NULL,NULL);
    clBuildProgram(program,num,devices,NULL,NULL,NULL);
    return 0;
}

//Funcion para no tener que estar pasando constantemente el kernel y el programa
float** matmul(float** mat1,float** mat2,int s1[],int s2[]){
    return matmul_cl(program,queue,context,mat1,mat2,s1,s2);
}

float dot_product(float* v1,float* v2,int s){
    return dot_product_cl(program,queue,context,v1,v2,s);
}


int main(){
    init_opencl();

    int s1[2] = {4,4};
    int s2[2] = {4,1};
    float** mat1 = make_random_matrix(s1,1);
    float** mat2 = make_random_matrix(s2,1);

    printf("Primer matriz \n");
    show_matrix(mat1,s1);
    printf("Segunda matriz \n");
    show_matrix(mat2,s2);
    
    float** matre = matmul(mat1,mat2,s1,s2);
    printf("Producto matricial\n");
    show_matrix(matre,(int[]){s1[0],s2[1]});
    
}