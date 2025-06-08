---
title: DCU 上的 Matrix Core 编程（part2）
authors:
  - genshen
tags: [DCU, hpc]
slug: /dcu-matrix-core-programming-part2
---

<!-- todo: 预备知识：调度 -->
## Matrix Core 上矩阵乘指令的基本格式

我们考虑矩阵乘：

$$D = A \times B + C$$

其中，A 矩阵是 $M \times K$ 的（M 行、K列，后续记号也是一样的含义）， B 矩阵是 $K \times N$的，
C、D 矩阵都是 $M \times N$ 的。

对应的 API 的格式为： `d = __builtin_amdgcn_mmac_CDFmt_MxNxKABFmt(a, b, c);`
其中，约定如下：
- ABFmt：$A$、$B$ 矩阵中元素的精度，可以是 f16、f32、f64、bf16、tf16、i8、u8 等。
- CDFmt：$C$、$D$ 矩阵中元素的精度，可以是 f16、f32、f64、bf16、tf16、i8、u8 等，如果 CDFmt 和ABFmt 一样，则省略 CDFmt。
- M、N、K：对应上面表述的 A、B、C、D 矩阵的行列数：
    - mA[M][K]: 输入矩阵A；
    - mB[K][N]: 输入矩阵B；
    - mC[M][N]: 累加输入矩阵C；
    - mD[M][N]: 累加结果矩阵D。
- a：向量寄存器（VGPRs），用于存储输入矩阵 A；
- b：向量寄存器（VGPRs），用于存储输入矩阵 B；
- c：向量寄存器（VGPRs），用于存储累加输入矩阵 C；
- d：向量寄存器（VGPRs），用于存储累加结果矩阵 D。

:::info
虽然 AMD 也有[类似的文档](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html)，但其指令格式和 DCU 上还是有很多差异（AMD 用的是 MFMA形式，如 `v_mfma_f32_16x16x4_f32`(AMD) vs `v_mmac_16x16x4_f32`(DCU) ），所以本教程适合中国宝宝。
:::

:::note
我们可以看到，相对于AMD 的指令，DCU 上的指令似乎参数要少。
:::

{/* truncate */}

## 示例：v_mmac_16x16x8_f32
<!-- todo: 明确调用的时钟周期 -->
这里，继续以 part1 的 `v_mmac_16x16x8_f32` 指令（对应API：`_builtin_amdgcn_mmac_f32_16x16x8f32`）为例，讲述 Matrix Core 中矩阵乘的执行过程。
因为了解到执行的细节后，才能方便针对架构特点和数据排布的特点开展实现和性能优化。

我们知道，DCU中是 64 个线程组成一个 wavefront（注：词 wavefront 和 warp 偶尔会混用；另：英伟达 GPU 上是 32个线程组成一个 warp）。

依据上面的规则，A、B矩阵分别是 $16 \times 8$ 和 $8\times 16$ 的，C、D 矩阵都是 $16 \times 16$ 大小的， A、B、C、D矩阵中的数据类型都是 f32。

wavefront 内的寄存器 layout 如下：
![](mt-16x16x8.png)

## 参考资料（AMD）
- https://fs.hlrs.de/projects/par/events/2024/GPU-AMD/day4/20.%20AMD_Matrix_Cores.pdf
- AMD matrix cores 博客: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html
- https://gpuopen.com/learn/wmma_on_rdna3/
