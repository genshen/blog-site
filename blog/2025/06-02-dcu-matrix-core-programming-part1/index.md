---
title: DCU 上的 Matrix Core 编程（part1）
authors:
  - genshen
tags: [DCU, hpc]
slug: /dcu-matrix-core-programming-part1
---


## 预备知识：Matrix Core 简介
英伟达GPU上，Volta架构首次引入 tensor core，可以在一个周期内计算完两个4*4矩阵的乘法，其最广泛的用法就是矩阵乘，因此tensor core硬件受到深度学习研究者的广泛欢迎。

AMD 的GPU，也自 CDNA架构开始引入 Matrix Core，来对标 NVIDIA Tensor Core。
由于功能类似，**所以有时候，我们也将 Matrix Core 称为 tensor core**，但这两者在编程接口上区别还是比较大的。
DCU 架构的GPU，由于部分采用ROCm生态，其上的 Matrix Core 编程接口和 AMD GPU相近，因此很多 AMD GPU的编程资料可供参考。

![](https://www.amd.com/content/dam/amd/en/videos/2325906-matrix-cores-web-02.mp4)

<video width="100%" height="auto" controls>
<source src="https://www.amd.com/content/dam/amd/en/videos/2325906-matrix-cores-web-02.mp4" type="video/mp4"/>
</video>

- AMD Matrix Core 文档：https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html  
- 另一个关于 Matrix Core 的介绍：https://leiblog.wang/High-Performance-AMD-Matrix-Core/

{/* truncate */}

## DCU 系列命名规则
本章节内容均来自于公开资料。

| DCU 名称 | 代号 | --arch| 性能 | 备注|
| -- | -- | -- | -- | -- |
| Z100 的前一代 | / | gfx906 | / | 约2019 年使用| 
| Z100 | 子房 | gfx906 | Flops 和带宽见这里: 文献1[^1], 文献2[^2] | / |
| K100 | 孔明 | gfx926 | Flops 提升 | 一些简单的 Matrix core 指令的支持 |
| K100AI | 孔明 | gfx928 | 无双精度支持 | 用于AI训练，不支持 FP64。多了些 Matrix core 指令支持。 |
| BW100  | 伯温  | gfx936 | Flops 提升，全精度 Matrix Core 支持 |下一代（2025年）DCU 卡|

<!--见昆山计算中心、中科院东方超算--> 

## 从 Hello World 开始
### 环境
在开始之前，我们先明确相关的环境：
- 编译环境采用 DTK 25.04（基于 clang 15.0.0, dcc 24.10.0-0）；
- 硬件包括：K100 和 K100-AI （之前的 Z100 由于对 Matrix Core 指令的支持较少）。
一般地，在超算互联网平台，可以通过如下的命令加载相应的 DTK 环境（实训环境中，可以直接使用，不用加载）：
```bash
module load compiler/dtk/25.04
```

### 编译运行 Hello World 的例子

然后，我们考虑如下的一个 Hello World 代码：
```cpp
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>

#include "hip/hip_runtime.h"

__global__ void 
vectoradd_float() {
  using float4 = __attribute__( (__vector_size__(4 * sizeof(float)) )) float;
  using float2 = __attribute__( (__vector_size__(2 * sizeof(float)) )) float;
  float4 dmn = {0, 0, 0, 0};
  float2 amk = {1.0, 1.0};
  float2 bkn = {1.0, 1.0};
  dmn = __builtin_amdgcn_mmac_f32_16x16x8f32(amk, bkn, dmn);      

  const int tid = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

  printf("tid: %d, value: %f %f %f %f.\n", tid, (float)dmn[0], (float)dmn[1], (float)dmn[2], (float)dmn[3]);
  return;
}

int main() {
  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  std::cout << " System minor " << devProp.minor << std::endl;
  std::cout << " System major " << devProp.major << std::endl;
  std::cout << " agent prop name " << devProp.name << std::endl;

  vectoradd_float<<<4, 256>>>();
  hipDeviceSynchronize();
  return 0;
}
```

我们将以上代码保存为 tc.main.cpp，并采用 dcc 编译器对其进行编译：
```bash
# gfx928 是 K100-AI 架构，如果是其他硬件，这个参数需要修改（参考本文最开始的表）。
dcc --offload-arch=gfx928 tc.main.cpp -o tc
```

然后，我们就可以运行代码了：
```bash
# 实训平台：可以直接运行：
./tc

# 超算互联网平台：需要用 srun 运行（队列名称需依据账号资源确定）：
srun -n 1 -p wzidnormal --gres=dcu:1 ./tc
```
如果正常，会有如下的结果：
```log
srun: job 20664849 queued and waiting for resources
srun: job 20664849 has been allocated resources
srun: ROUTE: split_hostlist: hl=xdb6 tree_width 0
 System minor 2
 System major 9
 agent prop name DCU K100_AI
tid: 448, value: 8.000000 8.000000 8.000000 8.000000.
tid: 449, value: 8.000000 8.000000 8.000000 8.000000.
tid: 450, value: 8.000000 8.000000 8.000000 8.000000.
tid: 451, value: 8.000000 8.000000 8.000000 8.000000.
tid: 452, value: 8.000000 8.000000 8.000000 8.000000.
tid: 453, value: 8.000000 8.000000 8.000000 8.000000.
tid: 454, value: 8.000000 8.000000 8.000000 8.000000.
tid: 455, value: 8.000000 8.000000 8.000000 8.000000.
tid: 456, value: 8.000000 8.000000 8.000000 8.000000.
tid: 457, value: 8.000000 8.000000 8.000000 8.000000.
...
<过多日志省略掉了>
```
可以看到，我们已经成功基于 Matrix Core 实现的 wavefront 级（或者 warp 级的）的矩阵乘法。

### 汇编代码
我们修改上面的 dcc 编译命令，添加 `--save-temps` 参数，输出 DCU 端的汇编代码。

```bash
dcc --save-temps --offload-arch=gfx928 tc.main.cpp -o tc
```
我们会看到这里会生成一个文件：`tc.main-hip-amdgcn-amd-amdhsa-gfx928.s`，打开即可看到相关的 Matrix Core 的汇编代码，例如：
```s lighlight={3}
s_add_u32 s0, s0, s17
v_mov_b32_e32 v25, v0
v_mmac_16x16x8_f32 v[3:6], v[7:8], v[7:8], v[3:6]
v_lshlrev_b32_e32 v0, 20, v2
v_lshlrev_b32_e32 v1, 10, v1
s_addc_u32 s1, s1, 0
s_mov_b32 s28, s14
```

### 代码的简单解释
对于有一定 CUDA、HIP 编程基础的人员，上面代码的基本形式（核函数、如何启动核函数）应该不会陌生。
但针对 Matrix Core 编程，这里引入了两个额外的新内容：
1. 向量扩展：这里可以参考 GCC的文档：https://gcc.gnu.org/onlinedocs/gcc/Vector-Extensions.html
2. builtin mmac 函数：用于实现对 matrix core 的调用，这部分在下一部分会详细介绍。

## CMake 构建系统对 Matrix Core 的支持
上述的编译命令，核心是`--offload-arch` 选项（类似于 CUDA 平台的 `-arch` 选项）。考虑到很多大型项目都采用 CMake 作为项目的构建工具，本小节将讨论如何利用 CMake 进行配置。

注：由于需要用 CMake 的HIP_ARCHITECTURES 特性，CMake 版本需要 3.21 及以上版本。

我们考虑以下的工程结构：
```log
./
├── CMakeLists.txt
├── library
│   ├── CMakeLists.txt
│   ├── libfoo.cpp
│   └── libfoo.h
└── main.cpp
```
其中，`main.cpp` 是入口，libaray 目录中的源代码将会被编译为一个库 libfoo.a，然后被 main.cpp 链接生成二进制代码。
此外，libaray 目录中的 `libfoo.cpp` 中包含核函数，且核函数为 Hello World 例子中的带有 Matrix Core 支持的核函数。

### CMakeLists.txt 文件的配置
首先，是根目录的下的 CMakeLists.txt：
```cmake
cmake_minimum_required(VERSION 3.21)

project(matrix-core-cmake-demo LANGUAGES HIP CXX)

set(CMAKE_CXX_STANDARD 11)

set(MY_LIB_NAME foo)
add_subdirectory(library)

add_executable(matrix-core-demo main.cpp)
target_link_libraries(matrix-core-demo PUBLIC ${MY_LIB_NAME})
```
由于 DTK 25.04 已经支持在 CMake 中将 HIP 作为一种语言使用了。
这里，`project()` 中直接指定了项目的语言为 HIP、CXX 两种。

其中，HIP 语言默认的源代码后缀是`.hip`，也可以通过 `set_source_files_properties` 指定其他文件名为 HIP 语言。最后，将 main.cpp 编译成可执行文件，并链接 library 下生成的库。

对于 library 目录下的 CMakeLists.txt，配置如下：
```cmake
add_library(${MY_LIB_NAME} libfoo.cpp)

set_source_files_properties(libfoo.cpp PROPERTIES LANGUAGE HIP)
set_property(TARGET ${MY_LIB_NAME} PROPERTY HIP_ARCHITECTURES gfx928) # for K100-AI 

target_include_directories(
            ${MY_LIB_NAME}
            PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
)
```
这里，利用 `set_source_files_properties` 将 `libfoo.cpp` 指定为 HIP 语言代码。

编译 CMake 工程：
```bash
cmake -B ./build -S ./ -DCMAKE_HIP_COMPILER=dcc -DCMAKE_CXX_COMPILER=hipcc
cmake --build ./build --verbose
```
这里，指定 dcc 是 HIP代码的编译器，hipcc 是 C++代码的编译器（C++代码中可能会有一些 HIP API的调用（如 hipMalloc），故采用 dcc 或者hipcc 进行编译）。

### 指定特定的架构
由于`libfoo.cpp` 中包含了针对 gfx928 架构的核函数代码，因此需要指定只针对该架构进行编译。这个可以通过 [HIP_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/HIP_ARCHITECTURES.html#prop_tgt:HIP_ARCHITECTURES) 实现：
```cmake
set_property(TARGET ${MY_LIB_NAME} PROPERTY HIP_ARCHITECTURES gfx928)
```
可以在构建阶段，用`cmake --build ./build --verbose` 命令输出编译命令，
对比下添加这句前后输出的编译命令的差异(删除线是之前的，默认包含了所有的架构)：
```diff
- clang++  -I"/work/home/genshen/tc2/library" --offload-arch=gfx906 --offload-arch=gfx926 --offload-arch=gfx928 --offload-arch=gfx936 -std=gnu++11 -o CMakeFiles/foo.dir/libfoo.cpp.o -x hip -c /work/home/genshen/tc2/library/libfoo.cpp
+ clang++  -I/work/home/genshen/tc2/library --offload-arch=gfx928 -std=gnu++11 -o CMakeFiles/foo.dir/libfoo.cpp.o -x hip -c /work/home/genshen/tc2/library/libfoo.cpp
```

<!-- todo: target `hip::host` 与 `hip::device` -->

关于 CMake 对 HIP 的支持，更多可参考 AMD 的文档 https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html （由于 DCU 生态和 AMD GPU生态不完全一致，不保证该文档上的内容完全适用）。


## 参考文献
[^1]: B. Chen, J. Li, P. Li, J. Xie, R. Zhao, W. Gao, and L. Han, “Distributed training optimization for dcu,” in 2024 5th International Conference on Information Science, Parallel and Distributed Systems (ISPDS), 2024, pp. 387–392. Available:https://doi.org/10.1109/ISPDS62779.2024.10667546  
[^2]: W. Fan, H. Hua, J. Shang, Z. Wen, H. Guo, and L. Zhang, “Optimizing 2D convolution for DCUs,” CCF Transactions on High Performance Computing, vol. 7, no. 2, pp. 142–154, Apr. 2025. [Online]. Available:https://doi.org/10.1007/s42514-024-00205-y
