__kernel void DotP(
        __global float* v1,
        __global float* v2,
        __global float* v1_v2){
    int gid = get_global_id(0);

    v1_v2[gid] = v1[gid]*v2[gid];

}

__kernel void AddMat(
        __global float* mat1,
        __global float* mat2,
        __global float* madd){
    
    int gid = get_global_id(0);
    madd[gid] = mat1[gid]+mat2[gid];
    

}