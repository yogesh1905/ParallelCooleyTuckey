#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <dirent.h>
#include <cstring>
#include<omp.h>
#include <math.h>
#include <time.h>

using namespace std;

#define N_REPEAT 3

// Complex numbers data type
typedef float2 Cplx;

// Complex numbers operations
static __device__ __host__ inline Cplx CplxAdd(Cplx a, Cplx b) {
  Cplx c; c.x = a.x + b.x; c.y = a.y + b.y; return c;
}

static __device__ __host__ inline Cplx CplxInv(Cplx a) {
  Cplx c; c.x = -a.x; c.y = -a.y; return c;
}

static __device__ __host__ inline Cplx CplxMul(Cplx a, Cplx b) {
  Cplx c; c.x = a.x * b.x - a.y + b.y; c.y = a.x * b.y + a.y * b.x; return c;
}

/**
 * Reorders array by bit-reversing the indexes.
 */
__global__ void bitrev_reorder(Cplx* __restrict__ r, Cplx* __restrict__ d, int s, size_t nthr) {
  int id = blockIdx.x * nthr + threadIdx.x;
  r[__brev(id) >> (32 - s)] = d[id];
}

/**
 * Inner part of FFT loop. Contains the procedure itself.
 */
__device__ void inplace_fft_inner(Cplx* __restrict__ r, int j, int k, int m, int n) {
  if (j + k + m / 2 < n) { 
    Cplx t, u;
    
    t.x = __cosf((2.0 * M_PI * k) / (1.0 * m));
    t.y = -__sinf((2.0 * M_PI * k) / (1.0 * m));
    
    u = r[j + k];
    t = CplxMul(t, r[j + k + m / 2]);

    r[j + k] = CplxAdd(u, t);
    r[j + k + m / 2] = CplxAdd(u, CplxInv(t));
  }
}

/**
 * Middle part of FFT for small scope paralelism.
 */
__global__ void inplace_fft(Cplx* __restrict__ r, int j, int m, int n, size_t nthr) {
  int k = blockIdx.x * nthr + threadIdx.x;
  inplace_fft_inner(r, j, k, m, n);
}

/**
 * Outer part of FFT for large scope paralelism.
 */
__global__ void inplace_fft_outer(Cplx* __restrict__ r, int m, int n, size_t nthr) {
  int j = (blockIdx.x * nthr + threadIdx.x) * m;
  
  for (int k = 0; k < m / 2; k++) {
    inplace_fft_inner(r, j, k, m, n);
  }
}

/**
 * Runs in-place Iterative Fast Fourier Transformation.
 */
void fft(Cplx* __restrict__ d, size_t n, size_t threads, int balance) {
  size_t data_size = n * sizeof(Cplx);
  Cplx *r, *dn;
  
  // Copy data to GPU
  cudaMalloc((void**)&r, data_size);
  cudaMalloc((void**)&dn, data_size);
  cudaMemcpy(dn, d, data_size, cudaMemcpyHostToDevice);
  
  // Bit-reversal reordering
  int temp=n;
  int s=0;
  while(temp>0)
  {
    temp/=2;
    s++;
  }
  // int s = log2(n);
  bitrev_reorder<<<ceil(n / threads), threads>>>(r, dn, s, threads);
  
  // Synchronize
  cudaDeviceSynchronize();
  
  // Iterative FFT (with loop paralelism balancing)
  for (int i = 1; i <= s; i++) {
    int m = 1 << i;
    if (n/m > balance) {
      inplace_fft_outer<<<ceil((float)n / m / threads), threads>>>(r, m, n, threads);
    } else {
      for (int j = 0; j < n; j += m) {
        float repeats = m / 2;
        inplace_fft<<<ceil(repeats / threads), threads>>>(r, j, m, n, threads);
      }
    }
  }
  
  // Copy data from GPU & free the memory blocks
  Cplx* result;
  result = (Cplx*)malloc(data_size / 2);
  cudaMemcpy(result, r, data_size / 2, cudaMemcpyDeviceToHost);
  cudaFree(r);
  cudaFree(dn);
}


int main(int argc, char** argv) {
  srand (time(NULL));
  int n;
  cin>>n;
  int len=pow(2,n);
  vector<Cplx> buffer;
  Cplx temp;
  
  for (int i = 0; i < len; ++i)
    {
      temp.x=i;
      temp.y=i+2;
      buffer.push_back(temp);
    }
  int threads=512;
  int bal=128;
  ofstream myfile;
  // string fname="parallel"+".txt";
  myfile.open("parallel.txt", std::ios_base::app);
  int ind=0;
  for (int z = 1; z <= len; z <<= 1) {
        
      ind+=1;
      // float start = omp_get_wtime();
      // Run FFT algorithm with loaded data
      clock_t t; 
      t = clock(); 
      
      fft(&buffer[0], z, threads, bal);
      
      t = clock() - t; 
      double time_taken = ((double)t)/CLOCKS_PER_SEC;
      myfile <<ind<<" "<<time_taken<<endl;
      cout<<"for n "<<z<<"time " << time_taken <<" seconds"<< endl; 
        
    }
    myfile.close();


  return 0;
}
