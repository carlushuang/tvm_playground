tvm_playground
==============

compile tvm+amdgpu from source
------------------------------

install rocm, https://rocm.github.io/ROCmInstall.html

following instruction from https://docs.tvm.ai/install/from_source.html
need download llvm to build amdgpu support http://releases.llvm.org/download.html, I use llvm-7(clang+llvm-7.0.0-x86_64-linux-gnu-ubuntu-16.04/)

in config.cmake, change set(USE_ROCM ON), set(USE_LLVM /path/to/llvm-config)

NOTICE: must install gcc>6 inorder to build tvm compatible with llvm-7. I use gcc-7
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-7 g++-7
```

then following instructions of tvm install. prefer to use a new conda environment

