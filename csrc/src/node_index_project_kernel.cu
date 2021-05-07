#include "cuda_common.h"
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
// #include <ATen/Error.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <iostream>
#define THREAD_NUM 1024
#define MAX_FACE_CANDIDATE_HYPOTHESIS 300
enum Resize_Type {BILINEAR,NEAREST};

__global__ void resize_mask_kernel(
    int* src,
    int* dst,
    int sh,
    int sw,
    int dh,
    int dw
){
    //int pixel_num = sh*sw; 
    //2D Index of current thread
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;

    float scale_x = (float)sw/(float)dw;
    float scale_y = (float)sh/(float)dh;
    //printf("h,w:%d,%d",sh,dh);
    //printf("h,w:%f,%f",scale_x,scale_y);
    
    float src_x = ((float)dx + 0.5) * scale_x - 0.5;
    float src_y = ((float)dy + 0.5) * scale_y - 0.5;

    //printf("x,y:%f,%f",src_x,src_y);


    if((dx < dw) && (dy < dh)){
        //printf("test:%d,%d\n",dh,dw);
        dst[dw*dy+dx] = src[sw * (int)src_y + (int)src_x];
        //printf("point:%d,%d\n",(int)src_y,(int)src_x);
    }
}

__global__ void resize_img_kernel(
    float* src,
    float* dst,
    int sh,
    int sw,
    int dh,
    int dw
)
{
    int pixel_num = sh*sw; 
    //2D Index of current thread
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;

    float scale_x = (float)sw/(float)dw;
    float scale_y = (float)sh/(float)dh;
    
    // float src_x = ((float)dx + 0.5) * scale_x - 0.5;
    // float src_y = ((float)dy + 0.5) * scale_y - 0.5;

    float src_x = (float)dx * scale_x;
    float src_y = (float)dy * scale_y;

    if((dx >= dw) || (dy >= dh)) return;
    
    
    int src_x_0 = int(floor(src_x));
    int src_y_0 = int(floor(src_y));
    int src_x_1 = (src_x_0 + 1) > (sw - 1) ? (sw-1):(src_x_0 + 1);
    int src_y_1 = (src_y_0 + 1) > (sh - 1) ? (sh-1):(src_y_0 + 1);


    
    for(int i=0;i<3;++i)
    {
        float value0 = ((float)src_x_1 - src_x) * src[i*pixel_num+sw*src_y_0+src_x_0] 
                + (src_x - (float)src_x_0) * src[i*pixel_num+sw*src_y_0+src_x_1];
        
        float value1 = ((float)src_x_1 - src_x) * src[i*pixel_num+sw*src_y_1+src_x_0]
                + (src_x - (float)src_x_0) * src[i*pixel_num+sw*src_y_1+src_x_1];
        
        dst[i*dh*dw+dw*dy+dx] = ((float)src_y_1 - src_y) * value0 
                                + (src_y - (float)src_y_0) * value1;
    }
    

        
}

__global__ 
void bbox_computation_kernel(
    int* face_idx,//[triangle_size,3]
    int* mesh_pts,//[Q,2]
    int* bbox,//[triangle_size,4]
    int triangle_size
){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(tid>=triangle_size) return;
    int x[3];
    int y[3];
    for(int i=0;i<3;++i)
    {
        int p = face_idx[tid*3+i];
        x[i] = mesh_pts[2*p];
        y[i] = mesh_pts[2*p+1];
    }
    int xmax =0;
    int ymax = 0;
    int xmin = 1000;
    int ymin = 1000;
    for(int j=0;j<3;++j)
    {
        if(x[j]>xmax) xmax = x[j];
        if(x[j]<xmin) xmin = x[j];
        if(y[j]>ymax) ymax = y[j];
        if(y[j]<ymin) ymin = y[j];
    }
    
    bbox[4*tid] = xmax;
    bbox[4*tid+1] = ymax;
    bbox[4*tid+2] = xmin;
    bbox[4*tid+3] = ymin;

}
__global__
void candidate_triangle_select_kernel(
    int* mask_u,//[mask_pts_num]
    int* mask_v,//[mask_pts_num]
    int* bbox,//[face_num,4]
    int* candidate_face,//[mask_pts_num,MAX_FACE_CANDIDATE_HYPOTHESIS]
    int* candidate_face_num,//[mask_pts_num]
    int mask_pts_num,
    int face_num
){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(tid>=mask_pts_num) return;
    //int nums = candidate_face_num[tid];
    
    
    for(int i=0;i<face_num;++i)
    {
        if(mask_u[tid]<=bbox[4*i]
            &&mask_u[tid]>=bbox[4*i+2]
            &&mask_v[tid]<=bbox[4*i+1]
            &&mask_v[tid]>=bbox[4*i+3]){
                
                candidate_face[MAX_FACE_CANDIDATE_HYPOTHESIS*tid+candidate_face_num[tid]] = i;
                candidate_face_num[tid] += 1;   

            }
    }
    
}


__global__ 
void triangle_validate_kernel(
    int* u,
    int* v,
    int* face_idx,
    int* mesh_projected_pts,
    float* mesh_projected_z,
    int* candidate_face_idx,
    int* candidate_face_num,
    int* result,
    int mask_pts_num,
    int face_num
){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx>=mask_pts_num) return;
    result[idx] = -1;
    float minimum_z = 1000.0;
    int final_face_idx = -1;

    for(int j =0; j<candidate_face_num[idx];++j)
    {
        
        int p0 = face_idx[candidate_face_idx[idx*MAX_FACE_CANDIDATE_HYPOTHESIS+j]*3];
        int p1 = face_idx[candidate_face_idx[idx*MAX_FACE_CANDIDATE_HYPOTHESIS+j]*3+1];
        int p2 = face_idx[candidate_face_idx[idx*MAX_FACE_CANDIDATE_HYPOTHESIS+j]*3 + 2];

        int m_x = mesh_projected_pts[2*p2] - mesh_projected_pts[2*p0];
        int m_y = mesh_projected_pts[2*p1] - mesh_projected_pts[2*p0]; 
        int m_z = mesh_projected_pts[2*p0] - u[idx];
        
        int n_x = mesh_projected_pts[2*p2+1] - mesh_projected_pts[2*p0+1];
        int n_y = mesh_projected_pts[2*p1+1] - mesh_projected_pts[2*p0+1];
        int n_z = mesh_projected_pts[2*p0+1] - v[idx];
        float x = m_y*n_z - n_y*m_z;
        float y = m_z*n_x - n_z*m_x;
        float z = m_x*n_y - n_x*m_y;
        
        float alpha = x / (z+1e-6);
        float beta = y / (z+1e-6);
        float gamma = 1.0 - (x+y) / (z+1e-6);
        bool a = (gamma>=0);
        bool b = (alpha>=0);
        bool c = (beta>=0);
        bool d = (abs(z)>=1.0);
        if(!(a&&b&&c&&d)) continue;
        
        float mean_z = (gamma*mesh_projected_z[p0] + alpha*mesh_projected_z[p1] + beta*mesh_projected_z[p2]);
        
        if(minimum_z > mean_z)
        {
            minimum_z = mean_z;
            final_face_idx = candidate_face_idx[idx*MAX_FACE_CANDIDATE_HYPOTHESIS+j];
        }
    }
   
    if(-1==final_face_idx) return;
    int p0 = face_idx[final_face_idx*3];
    int p1 = face_idx[final_face_idx*3+1];
    int p2 = face_idx[final_face_idx*3 + 2];
    int x0 = mesh_projected_pts[2*p0];
    int y0 = mesh_projected_pts[2*p0+1];
    int x1 = mesh_projected_pts[2*p1];
    int y1 = mesh_projected_pts[2*p1+1];
    int x2 = mesh_projected_pts[2*p2];
    int y2 = mesh_projected_pts[2*p2+1];
    int error0 = (u[idx] - x0)*(u[idx] - x0) +(v[idx] - y0)*(v[idx] - y0);
    int error1 = (u[idx] - x1)*(u[idx] - x1) +(v[idx] - y1)*(v[idx] - y1);
    int error2 = (u[idx] - x2)*(u[idx] - x2) +(v[idx] - y2)*(v[idx] - y2);

    if(error0<=error1&&error0<=error2)
    {
        result[idx] =p0;
    }
    else if(error1<=error0&&error1<=error2)
    {
        result[idx] =p1;
    }
    else
    {
        result[idx] =p2;
    }         
}


at::Tensor resize_mask(
    at::Tensor mat,
    int target_h,
    int target_w
){
    int w,h;
    h = mat.size(0);
    w = mat.size(1);

    int tdimx = 32;
    int tdimy = 32;
    int bdimx = target_w / 32;
    int bdimy = target_h / 32;

    dim3 bdim(bdimx,bdimy);
    dim3 tdim(tdimx,tdimy);

    auto dst = at::zeros({target_h,target_w},mat.options());
        resize_mask_kernel<<<bdim,tdim>>>(
            mat.data_ptr<int>(),
            dst.data_ptr<int>(),
            h,
            w,
            target_h,
            target_w
        );
    
    return dst;
}

at::Tensor resize_img(
    at::Tensor mat,
    int target_h,
    int target_w
){
    int w,h;

    h = mat.size(1);
    w = mat.size(2);

    int tdimx = 32;
    int tdimy = 32;
    int bdimx = target_w / 32;
    int bdimy = target_h / 32;
    dim3 bdim(bdimx,bdimy);
    dim3 tdim(tdimx,tdimy);

    auto dst = at::zeros({3,target_h,target_w},mat.options());
        resize_img_kernel<<<bdim,tdim>>>(
            mat.data_ptr<float>(),
            dst.data_ptr<float>(),
            h,
            w,
            target_h,
            target_w
        );

    return dst;
}

at::Tensor node_project(
    at::Tensor mask_u,
    at::Tensor mask_v,
    at::Tensor face_idx,
    at::Tensor mesh_projected_pts,
    at::Tensor mesh_projected_z
)
{
    int mask_pts_size = mask_u.size(0);
    int triangle_size = face_idx.size(0);
    int bdim0 = (triangle_size - 0.5) / THREAD_NUM + 1;
    int bdim1 = (mask_pts_size - 0.5) / THREAD_NUM + 1;
    int tdim = THREAD_NUM;
    auto candidate_face = at::zeros({mask_pts_size,MAX_FACE_CANDIDATE_HYPOTHESIS},mask_u.options());
    auto candidate_face_num = at::zeros({mask_pts_size},mask_u.options());
    auto bbox = at::zeros({triangle_size,4},face_idx.options());
    
    auto result = at::zeros({mask_pts_size},mask_u.options());

    bbox_computation_kernel<<<bdim0,tdim>>>(
        face_idx.data_ptr<int>(),
        mesh_projected_pts.data_ptr<int>(),
        bbox.data_ptr<int>(),
        triangle_size
    );

    candidate_triangle_select_kernel<<<bdim1,tdim>>>(
        mask_u.data_ptr<int>(),
        mask_v.data_ptr<int>(),
        bbox.data_ptr<int>(),
        candidate_face.data_ptr<int>(),
        candidate_face_num.data_ptr<int>(),
        mask_pts_size,
        triangle_size
    );

    triangle_validate_kernel<<<bdim1,tdim>>>(
        mask_u.data_ptr<int>(),
        mask_v.data_ptr<int>(),
        face_idx.data_ptr<int>(),
        mesh_projected_pts.data_ptr<int>(),
        mesh_projected_z.data_ptr<float>(),
        candidate_face.data_ptr<int>(),
        candidate_face_num.data_ptr<int>(),
        result.data_ptr<int>(),
        mask_pts_size,
        triangle_size
    );

    return result;
}