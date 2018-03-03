#include <vector>

#include "caffe/layers/geometric_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
//#include <iostream>

namespace caffe {

template <typename Dtype>
void GeometricLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  unsigned count = bottom[0]->count();
  caffe_gpu_sub(
        count,
        bottom[0]->gpu_data(),
        bottom[1]->gpu_data(),
        diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->shape(0) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
void GeometricLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const vector<int> shape = diff_.shape();
  //  N = shape(0)  C = shape(1)  H = shape(2)  W = shape(3)
  const unsigned spatial_dim = shape[2] * shape[3];
  const unsigned image_dim = shape[1] * spatial_dim;
  // Add errors to the fist independent dimension
  for( size_t i = 0; i < shape[0]; ++i ){
    for( size_t j = 0; j < num_dim_; ++j ){
      for( size_t k = 0; k < (shape[1]/num_dim_ - 1); ++k ){
        caffe_gpu_add(
          spatial_dim,
          diff_.gpu_data() + i*image_dim + j*spatial_dim,
          diff_.gpu_data() + i*image_dim + j*spatial_dim + (k+1)*num_dim_*spatial_dim,
          diff_.mutable_gpu_data() + i*image_dim + j*spatial_dim);
      }
    }
  }
  // Replace the error values in other dimensions
  for( size_t i = 0; i < shape[0]; ++i ){
    for( size_t j = 0; j < num_dim_; ++j ){
      for( size_t k = 0; k < (shape[1]/num_dim_ - 1); ++k ){
        caffe_gpu_memcpy(
          spatial_dim,
          diff_.gpu_data() + i*image_dim + j*spatial_dim,
          diff_.mutable_gpu_data() + i*image_dim + j*spatial_dim + (k+1)*num_dim_*spatial_dim);
      }
    }
  }
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GeometricLossLayer);

}
