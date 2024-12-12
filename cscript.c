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

//Recomendable cuando la matriz es cuadrada
float** scalar_cuad_mult(float** mat,float scalar,int s1[]){
    float** scalar_mat = make_diag_mat(s1[0],scalar);
    return matmul(scalar_mat,mat,s1,s1);
}

//Multiplicacion por escalar(en general)
float** scalar_mult(float** mat,int size[],float scalar){
    return scalar_mult_cl(program,queue,context,mat,size,scalar);
}
float** Add(float** mat1,float** mat2,int s1[]){
    return Add_cl(program,queue,context,mat1,mat2,s1,s1);
}

//Atensor porque yo lo hice, 
struct ATensor{
    float** y; //Valor Yi de salida
    float** dy; //Derivada de Yi con respecto a wij
};

//Hace el calculo de un modelo lineal
struct ATensor get_linear_m(float** x,float** w,float** b,int n_vars){
    int dim_w[2] = {n_vars,n_vars};
    int dim_a[2] = {n_vars,1};
    float** prod = matmul(w,x,dim_w,dim_a);
    float** matad = Add(prod,b,dim_a);
    float** grads = make_zero_mat(dim_w);
    for(int i=0;i<n_vars;i++){
        for(int j=0;j<n_vars;j++){
            grads[i][j] = x[j][0];
        }
    }
    struct ATensor output={matad,grads}; //Asignamos los valores
    return output;
}



int main(){
    init_opencl();

    int s1[2] = {2,2};
    int s2[2] = {2,1};
    
    float xa[] = {10,2};
    float** x = list2cmatrix(xa,2);
    float** w = make_random_matrix(s1,0);
    float** b = make_random_matrix(s2,1);

    struct ATensor output_model = get_linear_m(x,w,b,2);
    float** re = output_model.y;
    float** dre = output_model.dy;


    show_matrix(re,s2);
    printf("--------------------------------- \n");
    show_matrix(dre,s1);
}