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
typedef struct{
    float** y; //Valor Yi de salida
    float** dy; //Derivada de Yi con respecto a wij
    int shape[5]; //Dimension de la matriz Yi
    int dshape[5]; //Dimension de la matriz de derivadas
    int rank;
}ATensor;

void show_tensor(ATensor *tensor1){
    if(tensor1==NULL){
        printf("Error en el tensor \n");
        return;
    }
    if(tensor1->rank != 2){
        printf("El rango debe ser 2 \n");
        return;
    }
    show_matrix(tensor1->y,tensor1->shape);

}

ATensor make_random_tensor(int size[]){
    float** rtensor = make_random_matrix(size,1);
    ATensor ten1 = {rtensor};
    ten1.shape[0]=size[0];
    ten1.shape[1]=size[1];
    ten1.rank=2;
    return ten1;
}

//Hace el calculo de un modelo lineal
ATensor get_linear_m(float** x,float** w,float** b,int n_vars){
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
    ATensor output={matad,grads}; //Asignamos los valores
    output.shape[0] = dim_a[0];
    output.shape[1] = dim_a[1];
    output.dshape[0] = dim_w[0];
    output.dshape[1] = dim_w[1];
    return output;
}

//Sin paralelizar todavia
ATensor MeanSquaredError(ATensor* y_real, ATensor* y_fake){
    float** p_yreal = y_real->y;
    float** p_yfake = y_fake->y;
    
    //Error acumulado en cada batch, se va a promediar
    float* ac_loss = (float*)calloc(y_real->shape[0],sizeof(float));
    
    //Iteramos sobre cada batch
    for(int i=0;i<y_real->shape[0];i++){
        float loss_batch = 0;
        //Debemos iterar sobre cada elemento del batch
        for(int j=0;i<y_real->shape[1];i++){
            loss_batch += pow(p_yreal[i][j]-p_yfake[i][j],2);
        }
        ac_loss[i] = loss_batch;
    }
}


int main(){
    init_opencl();

    int s1[2] = {2000,2000};
    int s2[2] = {100,1};
    

    //float xa[] = {10,2};
    //float** x = list2cmatrix(xa,2);
    //float** w = make_random_matrix(s1,0);
    //float** b = make_random_matrix(s2,1);
    //printf("Matrices generadas \n");
    //ATensor output_model = get_linear_m(x,w,b,2);
    //show_tensor(&output_model);

    float** w = make_random_matrix_th(s1,1);
    show_matrix(w,s1);
}