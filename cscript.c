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
int main(){
    const char* KernelSource = readTextFile("kernel.cl");
    const cl_uint num = 1;
    cl_device_type devt = CL_DEVICE_TYPE_CPU;
    clGetDeviceIDs(NULL,devt,0,NULL,(cl_uint*)&num);
    cl_device_id devices[1];
    clGetDeviceIDs(NULL,devt,num,devices,NULL);

    cl_context context = clCreateContextFromType(NULL,devt,NULL,NULL,&cerror);
    if(cerror != CL_SUCCESS){
        printf("Error en el context \n");
    }
    clGetDeviceIDs(NULL,devt,1,devices,NULL);

    cl_command_queue queue = clCreateCommandQueueWithProperties(context,devices[0],NULL,&qerror);
    if(qerror != CL_SUCCESS){
        printf("Error en el comand queue \n");
    }
    cl_program program = clCreateProgramWithSource(context,1,(const char**)&KernelSource,NULL,NULL);
    clBuildProgram(program,num,devices,NULL,NULL,NULL);

    int s1[2] = {1000,1000};
    int s2[2] = {1000,1000};
    float** mat1 = make_random_matrix(s1,1);
    float** mat2 = make_random_matrix(s2,1);
    /*
    float** mat1 = make_zero_mat(s1);
    float** mat2 = make_zero_mat(s2);
    mat1[0][0]=-1;
    mat1[1][1]=-1;
    
    mat2[0][0]=-1;
    mat2[0][1]=5;
    mat2[1][0]=2;
    mat2[1][1]=1;
    */
    //show_matrix(mat2,s2);
    float** mat_mult = matmul_v2(program,queue,context,mat1,mat2,s1,s2);
    //show_matrix(mat_mult,s1);

}