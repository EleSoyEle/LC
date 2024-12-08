//Advertencia, este codigo es tan complejo porque esta optimizado para usar recursos del cpu y gpu en paralelo
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/opencl.h>
#include "utils.c"

//Se asume que los vectores tienen el mismo tamaño
float dot_product(cl_program program,cl_command_queue queue,cl_context context,float* v1,float* v2,int s){
    //Pasamos los dos vectores a la memoria del dispositivo
    cl_mem buffer_v1 = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*s,v1,NULL);
    cl_mem buffer_v2 = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*s,v2,NULL);
    //Creamos el vector con las multiplicaciones
    cl_mem mult_buffer = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(float)*s,NULL,NULL);

    cl_kernel kernel = clCreateKernel(program,"DotP",NULL);
    clSetKernelArg(kernel,0,sizeof(cl_mem),(void*)&buffer_v1);
    clSetKernelArg(kernel,1,sizeof(cl_mem),(void*)&buffer_v2);
    clSetKernelArg(kernel,2,sizeof(cl_mem),(void*)&mult_buffer);

    size_t global_size = s;
    clEnqueueNDRangeKernel(queue,kernel,1,NULL,&global_size,NULL,0,NULL,NULL);
    float* mult_ptr = (float*)calloc(s,sizeof(float));
    clEnqueueReadBuffer(queue,mult_buffer,CL_TRUE,0,sizeof(float)*s,mult_ptr,0,NULL,NULL);

    float dot_p = 0;
    for(int i=0;i<s;i++){
        dot_p += mult_ptr[i];
    }
    clReleaseMemObject(buffer_v1);
    clReleaseMemObject(buffer_v2);
    clReleaseMemObject(mult_buffer);
    clReleaseKernel(kernel);
    free(mult_ptr);
    return dot_p;
}

//Calcula la matriz transpuesta
float** Tmat(float** mat,int size[]){
    float** mat_trasp = (float**)calloc(size[1],sizeof(float*));
    for(int i=0;i<size[1];i++){
        mat_trasp[i] = (float*)calloc(size[0],sizeof(float));
        for(int j=0;j<size[0];j++){
            mat_trasp[i][j] = mat[j][i];
        }
    }
    return mat_trasp;
}


//s1:[m,k], s2:[j,n] ---> s1xs2:[m,n]
float** matmul(cl_program program,cl_command_queue queue,cl_context context,float** mat1,float** mat2,int s1[],int s2[]){
    float** matprod = (float**)calloc(s1[0],sizeof(float*));
    float** mat_transp = Tmat(mat2,s2);
    for(int i=0;i<s1[0];i++){
        matprod[i] = (float*)calloc(s2[1],sizeof(float));
        for(int j=0;j<s2[1];j++){
            matprod[i][j] = dot_product(program,queue,context,mat1[i],mat_transp[j],s1[1]);
        }
    }
    return matprod;
}

//Sin probar, seguramente no funciona
//Ya funciona y fue probada, dejo el comentario anterior para recordarlo con amor
cl_int kerr = CL_SUCCESS;
float** add(cl_program program,cl_command_queue queue,cl_context context,float** mat1, float** mat2, int s1[],int s2[]){

    float* m1 = twod2oned(mat1,s1);
    float* m2 = twod2oned(mat2,s2);

    cl_mem buff_mat1 = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*s1[0]*s1[1],m1,NULL);
    cl_mem buff_mat2 = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*s2[0]*s2[1],m2,NULL);
    cl_mem buff_madd = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(float)*s1[0]*s2[1],NULL,NULL);
    
    cl_kernel kernel = clCreateKernel(program,"AddMat",&kerr);
    if(kerr!=CL_SUCCESS){
        printf("Error al crear el kernel:%d \n",kerr);
    }
    clSetKernelArg(kernel,0,sizeof(cl_mem),(void*)&buff_mat1);
    clSetKernelArg(kernel,1,sizeof(cl_mem),(void*)&buff_mat2);
    clSetKernelArg(kernel,2,sizeof(cl_mem),(void*)&buff_madd);
    
    size_t iters = s1[0]*s1[1];
    clEnqueueNDRangeKernel(queue,kernel,1,NULL,&iters,NULL,0,NULL,NULL);
    float* added_arr = (float*)calloc(s1[0],sizeof(float)*s1[0]*s1[1]);
    clEnqueueReadBuffer(queue,buff_madd,CL_TRUE,0,sizeof(float)*s1[0]*s1[1],added_arr,0,NULL,NULL);
    float** added_mat = oned2twod(added_arr,s1[0]*s1[1],s1);
    
    //Borramos las cosas
    clReleaseKernel(kernel);
    clReleaseMemObject(buff_mat1);
    clReleaseMemObject(buff_mat2);
    clReleaseMemObject(buff_madd);
    free(added_arr);
    free(m1);
    free(m2);
    return added_mat;

}

float** make_identity(int dim){
    float** I = (float**)calloc(dim,sizeof(float*));
    for(int i=0;i<dim;i++){
        I[i] = (float*)calloc(dim,sizeof(float));
        I[i][i]=1;
    }
    return I;
}

//0:Distribucion uniforme
//1:Distribucion normal
#define sqpi sqrt(2*CL_M_1_PI)
float** make_random_matrix(int dim[2],int mode){
    float** rand_matrix = (float**)calloc(dim[0],sizeof(float*));
    srand((unsigned int)time(NULL));
    for(int i=0;i<dim[0];i++){
        rand_matrix[i] = (float*)calloc(dim[1],sizeof(float));
        for(int j=0;j<dim[1];j++){
            float rv = (float)rand()/(float)(RAND_MAX);
            if(mode==0){
                rv = expf(-pow(rv,2)/(2))/(sqpi);
            }
            rand_matrix[i][j] = rv;
        }
    }
    return rand_matrix;
}

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
        printf("Error en el comand queue \n");
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
    float** mat_sum = add(program,queue,context,mat1,mat2,s1,s2);
    //show_matrix(mat_sum,s1);

}