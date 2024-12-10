__kernel void DotP(
        __global float* v1,
        __global float* v2,
        __global float* v1_v2){
    int gid = get_global_id(0);

    v1_v2[gid] = v1[gid]*v2[gid];

}

__kernel void MatMul(
    __global float* mat1,
    __global float* mat2,
    __global float* matmul_re,
    __const int s1fil,
    __const int s1col,
    __const int s2fil,
    __const int s2col){
    

    int gid = get_global_id(0);
    
    int n = s1col; //Numero comun de filas y columnas
    
    int filas = s1fil;
    int columnas = s2col;
    float s = 0;
    int j = gid%columnas;
    int i = (gid-j)/columnas;
    
    
    for(int k=0;k<n;k++){
        s += mat1[(int)s1col*i+k]*mat2[(int)s2col*k+j];
    }
    
    matmul_re[gid]=s;
}

__kernel void AddMat(
    __global float* mat1,
    __global float* mat2,
    __global float* madd){


    int gid = get_global_id(0);
    madd[gid] = mat1[gid]+mat2[gid];
    

}

__kernel void SMult(
    __global float* mat,
    __global float* omat,
    __const float scalar){

    int gid = get_global_id(0);
    omat[gid] = scalar*mat[gid];
    }