#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>

namespace py = pybind11;

// =====================================================
// Tensor
// =====================================================

struct Tensor {
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<int> shape;

    Tensor(std::vector<int> s) {
        shape = s;
        int size = 1;
        for (int v : s) size *= v;
        data.resize(size, 0.0);
        grad.resize(size, 0.0);
    }
};

// =====================================================
// Utilities
// =====================================================

void zero_grad(Tensor& t){
    std::fill(t.grad.begin(), t.grad.end(), 0.0);
}

// void random_init(Tensor& t, double scale){
//     std::mt19937 gen(42);
//     std::uniform_real_distribution<> dist(-scale, scale);
//     for(size_t i=0;i<t.data.size();i++)
//         t.data[i] = dist(gen);
// }

void random_init(Tensor& t){
    static std::mt19937 gen(std::random_device{}());

    int fan_in = 1;
    for(int i=1;i<t.shape.size();i++)
        fan_in *= t.shape[i];

    double limit = std::sqrt(2.0 / fan_in);

    std::normal_distribution<> dist(0.0, limit);

    for(size_t i=0;i<t.data.size();i++)
        t.data[i] = dist(gen);
}




inline int idx2(int b,int f,int F){
    return b*F + f;
}

inline int idx4(int b,int c,int h,int w,
                int C,int H,int W){
    return b*C*H*W + c*H*W + h*W + w;
}

// =====================================================
// Conv2D
// =====================================================

Tensor conv_forward(Tensor& x,
                    Tensor& w,
                    Tensor& b){

    int B = x.shape[0];
    int C = x.shape[1];
    int H = x.shape[2];
    int W = x.shape[3];

    int F = w.shape[0];
    int K = w.shape[2];

    int outH = H - K + 1;
    int outW = W - K + 1;

    Tensor out({B, F, outH, outW});

    for(int bb=0;bb<B;bb++)
    for(int f=0;f<F;f++)
    for(int i=0;i<outH;i++)
    for(int j=0;j<outW;j++){

        double sum = b.data[f];

        for(int c=0;c<C;c++)
        for(int ki=0;ki<K;ki++)
        for(int kj=0;kj<K;kj++){

            int x_index = idx4(bb,c,i+ki,j+kj,C,H,W);
            int w_index = f*C*K*K + c*K*K + ki*K + kj;

            sum += x.data[x_index] * w.data[w_index];
        }

        out.data[idx4(bb,f,i,j,F,outH,outW)] = sum;
    }

    return out;
}

void conv_backward(Tensor& x,
                   Tensor& w,
                   Tensor& b,
                   Tensor& out){

    int B = x.shape[0];
    int C = x.shape[1];
    int H = x.shape[2];
    int W = x.shape[3];

    int F = w.shape[0];
    int K = w.shape[2];

    int outH = H - K + 1;
    int outW = W - K + 1;

    for(int bb=0;bb<B;bb++)
    for(int f=0;f<F;f++)
    for(int i=0;i<outH;i++)
    for(int j=0;j<outW;j++){

        double go = out.grad[idx4(bb,f,i,j,F,outH,outW)];
        b.grad[f] += go;

        for(int c=0;c<C;c++)
        for(int ki=0;ki<K;ki++)
        for(int kj=0;kj<K;kj++){

            int x_index = idx4(bb,c,i+ki,j+kj,C,H,W);
            int w_index = f*C*K*K + c*K*K + ki*K + kj;

            x.grad[x_index] += w.data[w_index] * go;
            w.grad[w_index] += x.data[x_index] * go;
        }
    }
}

// =====================================================
// ReLU
// =====================================================

Tensor relu_forward(Tensor& x){
    Tensor out(x.shape);
    for(size_t i=0;i<x.data.size();i++)
        out.data[i] = std::max(0.0, x.data[i]);
    return out;
}

void relu_backward(Tensor& x, Tensor& out){
    for(size_t i=0;i<x.data.size();i++)
        if(x.data[i] > 0)
            x.grad[i] += out.grad[i];
}

// =====================================================
// MaxPool 2x2
// =====================================================

Tensor maxpool_forward(Tensor& x){

    int B=x.shape[0];
    int C=x.shape[1];
    int H=x.shape[2];
    int W=x.shape[3];

    int outH=H/2;
    int outW=W/2;

    Tensor out({B,C,outH,outW});

    for(int b=0;b<B;b++)
    for(int c=0;c<C;c++)
    for(int i=0;i<outH;i++)
    for(int j=0;j<outW;j++){

        double m=-1e18;

        for(int di=0;di<2;di++)
        for(int dj=0;dj<2;dj++){

            int hi=i*2+di;
            int wj=j*2+dj;

            m = std::max(m,
                x.data[idx4(b,c,hi,wj,C,H,W)]);
        }

        out.data[idx4(b,c,i,j,C,outH,outW)] = m;
    }

    return out;
}

void maxpool_backward(Tensor& x,
                      Tensor& out){

    int B=x.shape[0];
    int C=x.shape[1];
    int H=x.shape[2];
    int W=x.shape[3];

    int outH=H/2;
    int outW=W/2;

    for(int b=0;b<B;b++)
    for(int c=0;c<C;c++)
    for(int i=0;i<outH;i++)
    for(int j=0;j<outW;j++){

        double go = out.grad[idx4(b,c,i,j,C,outH,outW)];

        double max_val=-1e18;
        int max_index=-1;

        for(int di=0;di<2;di++)
        for(int dj=0;dj<2;dj++){

            int hi=i*2+di;
            int wj=j*2+dj;

            int idx = idx4(b,c,hi,wj,C,H,W);

            if(x.data[idx] > max_val){
                max_val = x.data[idx];
                max_index = idx;
            }
        }

        if(max_index >= 0)
            x.grad[max_index] += go;
    }
}

// =====================================================
// Flatten
// =====================================================

Tensor flatten_forward(Tensor& x){

    int B = x.shape[0];
    int total = x.data.size()/B;

    Tensor out({B,total});
    out.data = x.data;
    return out;
}

void flatten_backward(Tensor& x, Tensor& out){

    size_t size = std::min(x.grad.size(), out.grad.size());

    for(size_t i=0;i<size;i++)
        x.grad[i] += out.grad[i];
}

// =====================================================
// Linear
// =====================================================

Tensor linear_forward(Tensor& x,
                      Tensor& w,
                      Tensor& b){

    int B=x.shape[0];
    int inF=w.shape[0];
    int outF=w.shape[1];

    Tensor out({B,outF});

    for(int bb=0;bb<B;bb++)
    for(int j=0;j<outF;j++){

        double sum=b.data[j];

        for(int i=0;i<inF;i++)
            sum+=x.data[idx2(bb,i,inF)]
                *w.data[i*outF+j];

        out.data[idx2(bb,j,outF)] = sum;
    }

    return out;
}

void linear_backward(Tensor& x,
                     Tensor& w,
                     Tensor& b,
                     Tensor& out){

    int B=x.shape[0];
    int inF=w.shape[0];
    int outF=w.shape[1];

    for(int bb=0;bb<B;bb++)
    for(int j=0;j<outF;j++){

        double go=out.grad[idx2(bb,j,outF)];
        b.grad[j]+=go;

        for(int i=0;i<inF;i++){
            x.grad[idx2(bb,i,inF)] += w.data[i*outF+j]*go;
            w.grad[i*outF+j] += x.data[idx2(bb,i,inF)]*go;
        }
    }
}

// =====================================================
// Cross Entropy
// =====================================================

double cross_entropy(Tensor& logits,
                     std::vector<int>& targets){

    int B=logits.shape[0];
    int C=logits.shape[1];

    double loss=0;

    for(int b=0;b<B;b++){

        double maxv=-1e18;
        for(int c=0;c<C;c++)
            maxv=std::max(maxv,
                logits.data[idx2(b,c,C)]);

        double sumexp=0;
        for(int c=0;c<C;c++)
            sumexp+=std::exp(
                logits.data[idx2(b,c,C)]-maxv);

        for(int c=0;c<C;c++){

            double soft=
              std::exp(logits.data[idx2(b,c,C)]-maxv)
              /sumexp;

            if(c==targets[b]){
                loss-=std::log(soft+1e-12);
                logits.grad[idx2(b,c,C)]
                    =(soft-1)/B;
            }
            else{
                logits.grad[idx2(b,c,C)]
                    =soft/B;
            }
        }
    }

    return loss/B;
}

// =====================================================
// SGD
// =====================================================

void sgd_update(Tensor& p,double lr){
    for(size_t i=0;i<p.data.size();i++){
        p.data[i]-=lr*p.grad[i];
        p.grad[i]=0.0;
    }
}

// =====================================================
// PYBIND
// =====================================================

PYBIND11_MODULE(backend,m){

    py::class_<Tensor>(m,"Tensor")
        .def(py::init<std::vector<int>>())
        .def_readwrite("data",&Tensor::data)
        .def_readwrite("grad",&Tensor::grad)
        .def_readwrite("shape",&Tensor::shape);

    m.def("conv_forward",&conv_forward);
    m.def("conv_backward",&conv_backward);
    m.def("relu_forward",&relu_forward);
    m.def("relu_backward",&relu_backward);
    m.def("maxpool_forward",&maxpool_forward);
    m.def("maxpool_backward",&maxpool_backward);
    m.def("flatten_forward",&flatten_forward);
    m.def("flatten_backward",&flatten_backward);
    m.def("linear_forward",&linear_forward);
    m.def("linear_backward",&linear_backward);
    m.def("cross_entropy",&cross_entropy);
    m.def("sgd_update",&sgd_update);
    m.def("zero_grad",&zero_grad);
    m.def("random_init",&random_init);
}
