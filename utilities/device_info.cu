#include <cuda.h>
#include <iostream>

int main(int argc, char const *argv[])
{
    // retrieve some info about the CUDA device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      std::cout << "Device Number: " << i << std::endl;
      std::cout << "  Device name: " << prop.name << std::endl;
      std::cout << "  max Blocks Per MultiProcessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
      std::cout << "  max Threads Per MultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
      std::cout << "  max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
      std::cout << "  num SM: " << prop.multiProcessorCount << std::endl;
      std::cout << "  num bytes sharedMem Per Block: " << prop.sharedMemPerBlock << std::endl;
      std::cout << "  num bytes sharedMem Per Multiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
      std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
      std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
      std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl << std::endl;
    }
    
    return 0;
}
