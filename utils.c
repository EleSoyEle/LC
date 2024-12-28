#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/opencl.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

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


//Se asume que los vectores tienen el mismo tamaño
float dot_product_cl(cl_program program,cl_command_queue queue,cl_context context,float* v1,float* v2,int s){
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

float** make_zero_mat(int dim[]){
    float** mat_zero = (float**)calloc(dim[0],sizeof(float*));
    for(int i=0;i<dim[0];i++){
        float* vec_zero = (float*)calloc(dim[1],sizeof(float));
        mat_zero[i]=vec_zero;
    }
    return mat_zero;
}




//Esta fue la primer implementacion de un producto matricial
//Esmuy lento, lo dejo para recordarlo con amor
//s1:[m,k], s2:[j,n] ---> s1xs2:[m,n]
float** matmul_slow(cl_program program,cl_command_queue queue,cl_context context,float** mat1,float** mat2,int s1[],int s2[]){
    float** matprod = (float**)calloc(s1[0],sizeof(float*));
    float** mat_transp = Tmat(mat2,s2);
    for(int i=0;i<s1[0];i++){
        matprod[i] = (float*)calloc(s2[1],sizeof(float));
        for(int j=0;j<s2[1];j++){
            matprod[i][j] = dot_product_cl(program,queue,context,mat1[i],mat_transp[j],s1[1]);
        }
    }
    return matprod;
}

//Misma estructura pero el modo de calculo distinto
cl_int kerr_mat = CL_SUCCESS;
cl_int error_buffer = CL_SUCCESS;
float** matmul_cl(cl_program program,cl_command_queue queue,cl_context context,float** mat1,float** mat2,int s1[],int s2[]){
    //Puntero donde vamos a alojar los resultados al final
    //float** matprod = make_zero_mat((float[2]){s1[1],s2[0]});
    float* m1 = twod2oned(mat1,s1);
    float* m2 = twod2oned(mat2,s2);
    int dim_mat_s = s1[0]*s2[1];
    cl_mem mat1_mem = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*s1[0]*s1[1],m1,NULL);
    cl_mem mat2_mem = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*s2[0]*s2[1],m2,NULL);
    cl_mem mat_mul = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(float)*dim_mat_s,NULL,NULL);

    cl_kernel kernel = clCreateKernel(program,"MatMul",&kerr_mat);

    if(kerr_mat != CL_SUCCESS){
        printf("Error al crear el kernel: %d \n",kerr_mat);
    }
    int s1fil = s1[0];
    int s1col = s1[1];
    int s2fil = s2[0];
    int s2col = s2[1];

    clSetKernelArg(kernel,0,sizeof(cl_mem),(void*)&mat1_mem);
    clSetKernelArg(kernel,1,sizeof(cl_mem),(void*)&mat2_mem);
    clSetKernelArg(kernel,2,sizeof(cl_mem),(void*)&mat_mul); 
    clSetKernelArg(kernel,3,sizeof(cl_int),&s1fil);
    clSetKernelArg(kernel,4,sizeof(cl_int),&s1col);
    clSetKernelArg(kernel,5,sizeof(cl_int),&s2fil);
    clSetKernelArg(kernel,6,sizeof(cl_int),&s2col);

    size_t iters = dim_mat_s;
    clEnqueueNDRangeKernel(queue,kernel,1,NULL,&iters,NULL,0,NULL,NULL);
    clFinish(queue);
    float* mat_mu_ptr_arr = (float*)calloc(dim_mat_s,sizeof(float));
    error_buffer = clEnqueueReadBuffer(queue,mat_mul,CL_TRUE,0,sizeof(float)*dim_mat_s,mat_mu_ptr_arr,0,NULL,NULL);
    if(error_buffer != CL_SUCCESS){
        printf("Error al leer el buffer");
    }
    float** matmul_re = oned2twod(mat_mu_ptr_arr,dim_mat_s,(int[2]){s1[0],s2[1]});

    clReleaseKernel(kernel);
    clReleaseMemObject(mat1_mem);
    clReleaseMemObject(mat2_mem);
    clReleaseMemObject(mat_mul);
    free(mat_mu_ptr_arr);
    free(m1);
    free(m2);
    return matmul_re;
}
//Sin probar, seguramente no funciona
//Ya funciona y fue probada, dejo el comentario anterior para recordarlo con amor
cl_int kerr = CL_SUCCESS;
float** Add_cl(cl_program program,cl_command_queue queue,cl_context context,float** mat1, float** mat2, int s1[],int s2[]){

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

float** make_diag_mat(int dim,float scalar){
    float** I = (float**)calloc(dim,sizeof(float*));
    for(int i=0;i<dim;i++){
        I[i] = (float*)calloc(dim,sizeof(float));
        I[i][i]=scalar;
    }
    return I;
}


typedef struct{
    int mode;
    float** rand_array;
    int d1;
    int d2;
    int id;
}ThreadRandomData;


#define sqpi sqrt(2*CL_M_1_PI)
#define PI 3.1415926535
void* MakeRandomList(void* vargp){
    ThreadRandomData* data_ac = (ThreadRandomData*)vargp;
    int mode = data_ac->mode;
    int myid = data_ac->id;
    int d1 = data_ac->d1;
    int d2 = data_ac->d2;

    data_ac->rand_array[myid] = (float*)calloc(d2,sizeof(float));
    for(int i=0;i<d2;i++){
        float rv = (float)rand()/(float)(RAND_MAX);
        if(mode==1){
            float rv2 = (float)rand()/(float)(RAND_MAX);
            //Esta formula nos permite pasar de una distribucion uniforme a una normal
            float z1 = sqrt(-2*log(rv))*cos(2*PI*rv2);
            rv = z1;
        }
        data_ac->rand_array[myid][i]=rv;
    }
}

float** make_random_matrix_th(int dim[2],int mode){
    float** rand_matrix = (float**)calloc(dim[0],sizeof(float*));
    srand((unsigned int)time(NULL));
    pthread_t threads[dim[0]];

    
    for(int i=0;i<dim[0];i++){
        ThreadRandomData* data = malloc(sizeof(ThreadRandomData));
        data->mode = mode;
        data->d1 = dim[0];
        data->d2 = dim[1];
        data->rand_array = rand_matrix;        
        data->id = i;
        pthread_create(&threads[i],NULL,MakeRandomList,(void *)data);

    }
    for(int i=0;i<dim[0];i++){
        pthread_join(threads[i],NULL);
    }
    
    return rand_matrix;
}


//0:Distribucion uniforme
//1:Distribucion normal
float** make_random_matrix(int dim[2],int mode){
    float** rand_matrix = (float**)calloc(dim[0],sizeof(float*));
    srand((unsigned int)time(NULL));
    for(int i=0;i<dim[0];i++){
        rand_matrix[i] = (float*)calloc(dim[1],sizeof(float));
        for(int j=0;j<dim[1];j++){
            float rv = (float)rand()/(float)(RAND_MAX);
            
            if(mode==1){
                float rv2 = (float)rand()/(float)(RAND_MAX);
                //Esta formula nos permite pasar de una distribucion uniforme a una normal
                float z1 = sqrt(-2*log(rv))*cos(2*PI*rv2);
                rv = z1;
            }
            rand_matrix[i][j] = rv;
        }
    }
    return rand_matrix;
}



float* list2ptr(float lista[],int size){
    float* ptr = (float*)calloc(size,sizeof(float));
    for(int i=0;i<size;i++){
        ptr[i]=lista[i];
    }
    return ptr;
}

float** list2cmatrix(float lista[],int size){
    float** matrix = (float**)calloc(size,sizeof(float*));
    for(int i=0;i<size;i++){
        float* ptr = (float*)calloc(1,sizeof(float));
        ptr[0] = lista[i];
        matrix[i]=ptr;
    }
    return matrix;
}


float** scalar_mult_cl(cl_program program,cl_command_queue queue,cl_context context,float** mat,int size[],float scalar){
    float* m1 = twod2oned(mat,size);
    int dim_mat = size[0]*size[1];
    cl_mem buff1 = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*dim_mat,m1,NULL);
    cl_mem buff2 = clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(float)*size[0]*size[1],NULL,NULL);
    
    cl_kernel kernel = clCreateKernel(program,"SMult",NULL);

    clSetKernelArg(kernel,0,sizeof(cl_mem),(void*)&buff1);
    clSetKernelArg(kernel,1,sizeof(cl_mem),(void*)&buff2);
    clSetKernelArg(kernel,2,sizeof(cl_float),&scalar);

    size_t calc_size = size[0]*size[1];
    clEnqueueNDRangeKernel(queue,kernel,1,NULL,&calc_size,NULL,0,NULL,NULL);
    float* scalar_mult_1d = (float*)calloc(dim_mat,sizeof(float));
    clEnqueueReadBuffer(queue,buff2,CL_TRUE,0,sizeof(float)*dim_mat,scalar_mult_1d,0,NULL,NULL);
    float** scalar_mult_2d = oned2twod(scalar_mult_1d,dim_mat,size);
    return scalar_mult_2d;
}