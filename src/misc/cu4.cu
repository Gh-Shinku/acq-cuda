#include "utils.hpp"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

__global__ void blurKernel(uint8_t *in, uint8_t *out, int w, int h,
                           int blurSize) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < h && col < w) {
    int pixelSum = 0;
    int pixelNum = 0;
    for (int blurRow = -blurSize; blurRow < blurSize + 1; ++blurRow) {
      for (int blurCol = -blurSize; blurCol < blurSize + 1; ++blurCol) {
        const int curRow = row + blurRow;
        const int curCol = col + blurCol;
        if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
          pixelSum += in[curRow * w + curCol];
          ++pixelNum;
        }
      }
    }
    out[row * w + col] = (uint8_t)(pixelSum / pixelNum);
  }
}

int main() {

  // CudaMemory<uint8_t> in();
}