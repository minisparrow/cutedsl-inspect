#!/usr/bin/env python
# coding: utf-8

"""
Test the Cutlass compiler with hello_world example
"""

import sys

import cutlass
import cutlass.cute as cute


@cute.kernel
def kernel():
    # Get the x component of the thread index (y and z components are unused)
    tidx, _, _ = cute.arch.thread_idx()
    # Only the first thread (thread 0) prints the message
    if tidx == 0:
        cute.printf("Hello world from device\n")


@cute.jit
def hello_world():
    # Print hello world from host code
    cute.printf("hello world from host\n")

    # Initialize CUDA context for launching a kernel with error checking
    cutlass.cuda.initialize_cuda_context()

    # Launch kernel
    kernel().launch(
        grid=(1, 1, 1),   # Single thread block
        block=(32, 1, 1)  # One warp (32 threads) per thread block
    )


if __name__ == "__main__":
    # Method 1: Just-In-Time (JIT) compilation - compiles and runs the code immediately
    hello_world()

    # Method 2: Compile first (useful if you want to run the same code multiple times)
    print("\nCompiling...")
    hello_world_compiled = cute.compile(hello_world)
    # Run the pre-compiled version
    print("Running compiled version...\n")
    hello_world_compiled()
    
    print("\nDone!")
