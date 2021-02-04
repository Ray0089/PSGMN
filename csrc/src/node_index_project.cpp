#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <iostream>


extern THCState* state;

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

using namespace std;

at::Tensor node_project(
    at::Tensor mask_u,
    at::Tensor mask_v,
    at::Tensor face_idx,
    at::Tensor mesh_projected_pts,
    at::Tensor mesh_projected_z
);
at::Tensor resize_mask(
    at::Tensor mat,
    int target_h,
    int target_w
);
at::Tensor resize_img(
    at::Tensor mat,
    int target_h,
    int target_w
);

at::Tensor node_project_launch(
    at::Tensor mask_u,
    at::Tensor mask_v,
    at::Tensor face_idx,
    at::Tensor mesh_projected_pts,
    at::Tensor mesh_projected_z
){
    auto result = node_project(
    mask_u,
    mask_v,
    face_idx,
    mesh_projected_pts,
    mesh_projected_z);
    return result;

}

at::Tensor resize_mask_launch(
    at::Tensor mat,
    int target_h,
    int target_w
){
    CHECK_INPUT(mat);
    auto result = resize_mask(
    mat,
    target_h,
    target_w);
    return result;

}

at::Tensor resize_img_launch(
    at::Tensor mat,
    int target_h,
    int target_w
){
    auto result = resize_img(
    mat,
    target_h,
    target_w);
    return result;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("node_project", &node_project_launch, "project mesh node to image pixels");
    m.def("resize_mask",&resize_mask_launch,"resize a 2d array");
    m.def("resize_img",&resize_img_launch,"resize a 2d array");
}