#include <vector>

#include "caffe/layers/geometric_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template<typename Dtype>
void GeometricLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // Number of independent dimentions. Default = 1
  num_dim_ = 1;
  bool has_num_dim = this->layer_param_.geometric_loss_param().has_num_dim();
  if (has_num_dim){
    int num_dim = this->layer_param_.geometric_loss_param().num_dim();
    CHECK_GT(num_dim, 0)
        << "Number of independent dimensions must be positive.";
    num_dim_ = num_dim;
  }
}

template <typename Dtype>
void GeometricLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GeometricLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  unsigned count = bottom[0]->count();
  caffe_sub( count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
        diff_.mutable_cpu_data() );
  caffe_abs( count, diff_.cpu_data(), diff_.mutable_cpu_data() );
  const vector<int> shape = diff_.shape();
//  N = shape(0)  C = shape(1)  H = shape(2)  W = shape(3)
  const unsigned spatial_dim = shape[2] * shape[3];
  const unsigned image_dim = shape[1] * spatial_dim;
  // Add errors to the fist independent dimensions
  for( size_t i = 0; i < shape[0]; ++i ){
    for( size_t j = 0; j < (shape[1]/num_dim_ - 1); ++j ){
      caffe_add(
        spatial_dim,
        diff_.cpu_data() + i*image_dim,
        diff_.cpu_data() + i*image_dim + (j+1)*num_dim_*spatial_dim,
        diff_.mutable_cpu_data() + i*image_dim);
    }
  }
  // Replace the error values in other dimensions
  for( size_t i = 0; i < shape[0]; ++i ){
    for( size_t j = 0; j < (shape[1]/num_dim_ - 1); ++j ){
      caffe_copy(
        spatial_dim,
        diff_.cpu_data() + i*image_dim,
        diff_.mutable_cpu_data() + i*image_dim + (j+1)*num_dim_*spatial_dim);
    }
  }
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->shape(0) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void GeometricLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->shape(0);
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GeometricLossLayer);
#endif

INSTANTIATE_CLASS(GeometricLossLayer);
REGISTER_LAYER_CLASS(GeometricLoss);

}
