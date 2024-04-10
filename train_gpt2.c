/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#ifdef OMP
#include <omp.h>
#endif

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes

/**
 * 前向编码器函数
 *
 * 本函数用于计算输入序列的编码表示。对每个位置的输入token，它将通过加权求和的方式，
 * 结合token嵌入（wte）和位置嵌入（wpe）来生成最终的输出。
 *
 * @param out 输出数组，形状为[B, T, C]，其中B为批大小，T为序列长度，C为嵌入维度。
 * @param inp 输入整数数组，形状为[B, T]，表示每个位置的token索引。
 * @param wte Token嵌入权重矩阵，形状为[V, C]，其中V为词汇表大小，C为嵌入维度。
 * @param wpe 位置嵌入权重矩阵，形状为[T, C]，其中T为序列长度，C为嵌入维度。
 * @param B 批大小。
 * @param T 序列长度。
 * @param C 嵌入维度。
 */
void encoder_forward(float *out,
                     int *inp, float *wte, float *wpe,
                     int B, int T, int C)
{
    for (int b = 0; b < B; b++)
    { // 遍历批中的每个样本
        for (int t = 0; t < T; t++)
        { // 遍历每个样本中的每个位置
            // 根据样本索引b和位置索引t，计算在输出数组out中的起始位置
            float *out_bt = out + b * T * C + t * C;
            // 根据输入数组inp中的值，获取当前位置的token索引
            int ix = inp[b * T + t];
            // 根据获取的token索引，定位到wte中对应的嵌入向量
            float *wte_ix = wte + ix * C;
            // 根据当前位置t，定位到wpe中对应的位置嵌入向量
            float *wpe_t = wpe + t * C;
            // 将token嵌入向量和位置嵌入向量相加，并存储到输出数组中
            for (int i = 0; i < C; i++)
            {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

/**
 * @brief 实现编码器的反向传播
 *
 * 该函数针对一个批次的输入数据，计算其在编码器层的反向传播误差，更新编码器的权重和偏差。
 *
 * @param dwte 指向编码器权重误差数组的指针，用于存储权重的反向传播误差。
 * @param dwpe 指向编码器偏差误差数组的指针，用于存储偏差的反向传播误差。
 * @param dout 指向输出层的梯度数组的指针。
 * @param inp 指向输入索引数组的指针，记录了每个token的输入特征索引。
 * @param B 批次大小。
 * @param T token数。
 * @param C 特征通道数。
 */
void encoder_backward(float *dwte, float *dwpe,
                      float *dout, int *inp,
                      int B, int T, int C)
{
    // 遍历批次中的每个样本
    for (int b = 0; b < B; b++)
    {
        // 遍历每个token
        for (int t = 0; t < T; t++)
        {
            // 计算当前token输出梯度的起始位置
            float *dout_bt = dout + b * T * C + t * C;
            // 获取当前token的输入特征索引
            int ix = inp[b * T + t];
            // 计算当前输入特征索引对应的权重误差起始位置
            float *dwte_ix = dwte + ix * C;
            // 计算当前token的偏差误差位置
            float *dwpe_t = dwpe + t * C;
            // 遍历每个特征通道，更新权重和偏差的误差
            for (int i = 0; i < C; i++)
            {
                // 获取当前特征通道的梯度
                float d = dout_bt[i];
                // 更新权重误差
                dwte_ix[i] += d;
                // 更新偏差误差
                dwpe_t[i] += d;
            }
        }
    }
}

/**
 * 层归一化前向传播函数
 *
 * 本函数对输入的特征数据进行层归一化处理，具体包括计算输入数据的均值、标准差，
 * 对输入数据进行归一化以及缩放和平移操作。该操作常用于神经网络中的残差块中，
 * 有助于稳定训练过程。
 *
 * @param out 输出数据数组，大小为BxTxC，其中B为批次大小，T为token（或序列长度），C为特征维度
 * @param mean 计算得到的均值数组，大小为BxT，用于反向传播
 * @param rstd 计算得到的标准差的倒数（即1/标准差），大小为BxT，用于反向传播
 * @param inp 输入数据数组，大小为BxTxC
 * @param weight 权重数组，用于对归一化后的数据进行缩放，大小为C
 * @param bias 偏置数组，用于对归一化后的数据进行平移，大小为C
 * @param B 批次大小
 * @param T token（或序列长度）
 * @param C 特征维度
 */
void layernorm_forward(float *out, float *mean, float *rstd,
                       float *inp, float *weight, float *bias,
                       int B, int T, int C)
{
    float eps = 1e-5f; // 用于防止除以零的小值

    for (int b = 0; b < B; b++)
    { // 遍历批次
        for (int t = 0; t < T; t++)
        { // 遍历token
            // 计算当前批次和token的输入特征起始位置
            float *x = inp + b * T * C + t * C;
            // 计算当前token的均值
            float m = 0.0f;
            for (int i = 0; i < C; i++)
            {
                m += x[i];
            }
            m /= C; // 平均值

            // 计算当前token的方差
            float v = 0.0f;
            for (int i = 0; i < C; i++)
            {
                float xshift = x[i] - m; // 去均值操作
                v += xshift * xshift;
            }
            v /= C; // 计算样本方差
            // 计算标准差的倒数
            float s = 1.0f / sqrtf(v + eps);

            // 计算输出特征的起始位置
            float *out_bt = out + b * T * C + t * C;
            // 对每个特征进行归一化、缩放和偏移
            for (int i = 0; i < C; i++)
            {
                float n = (s * (x[i] - m));        // 归一化
                float o = n * weight[i] + bias[i]; // 缩放和平移
                out_bt[i] = o;                     // 保存输出
            }
            // 缓存均值和标准差的倒数，以备反向传播使用
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

/**
 * 对输入数据进行层归一化反向传播计算。
 *
 * @param dinp 输入数据的梯度数组，输出梯度将存储在此数组中。
 * @param dweight 权重参数的梯度数组，归一化过程中使用的权重的梯度将存储在此数组中。
 * @param dbias 偏置参数的梯度数组，归一化过程中使用的偏置的梯度将存储在此数组中。
 * @param dout 输出数据的梯度数组，即前向传播时的输出的梯度。
 * @param inp 输入数据数组。
 * @param weight 归一化时使用的权重参数数组。
 * @param mean 前向传播时计算的均值数组。
 * @param rstd 前向传播时计算的标准差的倒数数组。
 * @param B 批量大小。
 * @param T token长（序列长度）。
 * @param C 特征通道数量。
 *
 * 此函数对输入数据进行层归一化操作的反向传播，计算输入数据、权重、偏置的梯度。
 */
void layernorm_backward(float *dinp, float *dweight, float *dbias,
                        float *dout, float *inp, float *weight, float *mean, float *rstd,
                        int B, int T, int C)
{
    for (int b = 0; b < B; b++)
    { // 遍历批量数据
        for (int t = 0; t < T; t++)
        { // 遍历token长
            float *dout_bt = dout + b * T * C + t * C;
            float *inp_bt = inp + b * T * C + t * C;
            float *dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // 首先进行两个减少操作，计算归一化梯度的平均值和加权平均值
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++)
            {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // 再次遍历，累积所有梯度
            for (int i = 0; i < C; i++)
            {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // 对偏置的梯度贡献
                dbias[i] += dout_bt[i];
                // 对权重的梯度贡献
                dweight[i] += norm_bti * dout_bt[i];
                // 对输入的梯度贡献
                float dval = 0.0f;
                dval += dnorm_i;                    // 第一项
                dval -= dnorm_mean;                 // 第二项
                dval -= norm_bti * dnorm_norm_mean; // 第三项
                dval *= rstd_bt;                    // 最终缩放
                dinp_bt[i] += dval;
            }
        }
    }
}

/**
 * 执行矩阵乘法前向传播操作。
 *
 * @param out 输出矩阵，尺寸为(B, T, OC)，其中B为批次大小，T为token数，OC为输出通道数。
 * @param inp 输入矩阵，尺寸为(B, T, C)，其中C为输入通道数。
 * @param weight 权重矩阵，尺寸为(OC, C)，表示从输入通道到输出通道的权重。
 * @param bias 偏置向量，尺寸为(OC)，为每个输出通道提供偏置值。如果为NULL，则不应用偏置。
 * @param B 批次大小。
 * @param T token数。
 * @param C 输入通道数。
 * @param OC 输出通道数。
 *
 * 该函数通过在输入矩阵和权重矩阵之间进行矩阵乘法，再加上可选的偏置项，来计算输出矩阵。
 * 使用OpenMP并行化来加速循环的执行。
 */
void matmul_forward(float *out,
                    float *inp, float *weight, float *bias,
                    int B, int T, int C, int OC)
{
// most of the running time is spent here and in matmul_backward
// OC is short for "output channels"
// inp is (B,T,C), weight is (OC, C), bias is (OC)
// out will be (B,T,OC)
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    { // 遍历批次
        for (int t = 0; t < T; t++)
        {                                              // 遍历token
            float *out_bt = out + b * T * OC + t * OC; // 计算当前token和批次的输出起始指针
            float *inp_bt = inp + b * T * C + t * C;   // 计算当前token和批次的输入起始指针
            for (int o = 0; o < OC; o++)
            {                                                // 遍历输出通道
                float val = (bias != NULL) ? bias[o] : 0.0f; // 如果有偏置，则初始化为偏置值，否则为0
                float *wrow = weight + o * C;                // 计算当前输出通道的权重行起始指针
                for (int i = 0; i < C; i++)
                { // 遍历输入通道，计算输出值
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val; // 将计算得到的值存入输出矩阵
            }
        }
    }
}

/**
 * matmul_backward函数用于矩阵乘法的反向传播计算。
 *
 * @param dinp 指向输入梯度的指针。
 * @param dweight 指向权重梯度的指针。
 * @param dbias 指向偏置梯度的指针，如果不需要计算偏置梯度，可以为NULL。
 * @param dout 指向输出梯度的指针。
 * @param inp 指向输入数据的指针。
 * @param weight 指向权重的指针。
 * @param B 批量大小。
 * @param T token长。
 * @param C 输入通道数。
 * @param OC 输出通道数。
 *
 * 此函数首先对输入数据(inp)进行反向传播计算，然后对权重(weight)和偏置(bias)进行更新。
 * 使用OpenMP并行化策略以提高计算效率。
 */
void matmul_backward(float *dinp, float *dweight, float *dbias,
                     float *dout, float *inp, float *weight,
                     int B, int T, int C, int OC)
{
// 在此处和matmul_forward中花费了大部分运行时间。
// 可以在一个“轮次”中完成反向传播的循环，但那不利于有效的并行化策略。

// 首先对输入(inp)进行反向传播，平行计算每个批量和token长。
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float *dout_bt = dout + b * T * OC + t * OC; // 计算输出梯度的指针。
            float *dinp_bt = dinp + b * T * C + t * C;   // 计算输入梯度的指针。
            for (int o = 0; o < OC; o++)
            {
                float *wrow = weight + o * C; // 指向当前输出通道权重行的指针。
                float d = dout_bt[o];         // 当前输出通道的梯度。
                for (int i = 0; i < C; i++)
                {
                    dinp_bt[i] += wrow[i] * d; // 计算输入梯度。
                }
            }
        }
    }
// 对权重和偏置进行反向传播，平行计算每个输出通道(OC)。
#pragma omp parallel for
    for (int o = 0; o < OC; o++)
    {
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *dout_bt = dout + b * T * OC + t * OC; // 计算输出梯度的指针。
                float *inp_bt = inp + b * T * C + t * C;     // 指向当前token长和批量的输入数据。
                float *dwrow = dweight + o * C;              // 指向当前输出通道权重梯度的指针。
                float d = dout_bt[o];                        // 当前输出通道的梯度。
                if (dbias != NULL)
                {
                    dbias[o] += d;
                } // 如果计算偏置梯度，则更新偏置梯度。
                for (int i = 0; i < C; i++)
                {
                    dwrow[i] += inp_bt[i] * d; // 计算权重梯度。
                }
            }
        }
    }
}

/**
 * 前向注意力机制计算函数
 *
 * 本函数用于计算输入数据的注意力权重，并根据权重对输入进行加权求和，生成输出。
 *
 * @param out 输出数组，尺寸为(B, T, C)，其中B为批次大小，T为序列长度，C为特征维度。
 * @param preatt 预计算的注意力得分，用于后续计算，尺寸为(B, NH, T, T)。
 * @param att 计算后的注意力得分，反映输入序列中各位置之间的关系，尺寸同preatt。
 * @param inp 输入数据，包含查询向量、键向量和值向量，尺寸为(B, T, 3C)。
 * @param B 批次大小。
 * @param T 序列长度。
 * @param C 特征维度，输入数据的特征维度。
 * @param NH 头数，即注意力头的数量。
 */
void attention_forward(float *out, float *preatt, float *att,
                       float *inp,
                       int B, int T, int C, int NH)
{
    // 计算每个头的特征尺寸
    int C3 = C * 3;
    int hs = C / NH;               // 每个头的尺寸
    float scale = 1.0 / sqrtf(hs); // 标准化缩放因子

#pragma omp parallel for collapse(3)
    // 遍历批次中的每个样本、token和头部
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < NH; h++)
            {
                // 计算当前token和头部对应的查询向量地址
                float *query_t = inp + b * T * C3 + t * C3 + h * hs;
                // 计算预计算注意力得分和最终注意力得分的地址
                float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                float *att_bth = att + b * NH * T * T + h * T * T + t * T;

                // 计算查询-键的点积并应用缩放因子，更新预计算注意力得分
                float maxval = -10000.0f; // 初始化最大值，用于后续softmax计算
                for (int t2 = 0; t2 <= t; t2++)
                {
                    // 计算键向量地址
                    float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;

                    // 计算点积并缩放
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++)
                    {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval)
                    {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // 计算exp并累加，用于softmax归一化
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // 归一化得到注意力权重
                for (int t2 = 0; t2 < T; t2++)
                {
                    if (t2 <= t)
                    {
                        att_bth[t2] *= expsum_inv;
                    }
                    else
                    {
                        // 应用因果掩码，确保只有当前及之前的序列位置影响当前位置的输出
                        att_bth[t2] = 0.0f;
                    }
                }

                // 根据注意力权重加权求和，生成输出
                float *out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++)
                {
                    out_bth[i] = 0.0f;
                }
                for (int t2 = 0; t2 <= t; t2++)
                {
                    // 计算值向量地址
                    float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++)
                    {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

/**
 * 注意力机制的反向传播函数
 *
 * @param dinp 输入数据的梯度，尺寸为(B, T, 3C)，其中Q、K、V的组合
 * @param dpreatt 关于预注意力的梯度，尺寸为(B, NH, T, T)
 * @param datt 关于注意力权重的梯度，尺寸为(B, NH, T, T)
 * @param dout 输出数据的梯度，尺寸为(B, T, C)
 * @param inp 输入数据，尺寸为(B, T, 3C)，其中Q、K、V的组合
 * @param att 注意力权重，尺寸为(B, NH, T, T)
 * @param B 批量大小
 * @param T token数
 * @param C 特征通道数
 * @param NH 头数，即注意力头的数量
 *
 * 本函数执行注意力机制的反向传播过程，计算输入数据、预注意力、注意力权重等参数的梯度。
 * 对于每个位置的每个头，遍历计算过程包括：值的累加、softmax、查询@键矩阵乘法的反向传播。
 */
void attention_backward(float *dinp, float *dpreatt, float *datt,
                        float *dout, float *inp, float *att,
                        int B, int T, int C, int NH)
{
    int C3 = C * 3;
    int hs = C / NH;               // 每个头的大小
    float scale = 1.0 / sqrtf(hs); // 缩放因子

    // 遍历批量、token数和头数
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < NH; h++)
            {
                // 计算各个变量的偏导地址
                float *att_bth = att + b * NH * T * T + h * T * T + t * T;
                float *datt_bth = datt + b * NH * T * T + h * T * T + t * T;
                float *dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
                float *dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float *query_t = inp + b * T * C3 + t * C3 + h * hs;

                // 反向传播值的累加
                float *dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                    float *dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                    for (int i = 0; i < hs; i++)
                    {
                        // 更新梯度
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // 反向传播softmax部分
                for (int t2 = 0; t2 <= t; t2++)
                {
                    for (int t3 = 0; t3 <= t; t3++)
                    {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // 反向传播查询@键矩阵乘法
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    float *dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C;
                    for (int i = 0; i < hs; i++)
                    {
                        // 更新梯度
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

/**
 * GELU激活函数的前向计算
 *
 * 本函数实现了GELU（Gaussian Error Linear Unit）激活函数的前向计算，GELU是一种在神经网络中使用的激活函数，
 * 相比于其他激活函数（如ReLU），它能提供更平滑的梯度，有助于避免梯度消失问题。
 *
 * @param out 输出数组，存储计算后的GELU激活值。
 * @param inp 输入数组，存储待计算的原始神经元值。
 * @param N 数组的大小，表示输入输出数组中的元素数量。
 */
void gelu_forward(float *out, float *inp, int N)
{
    // 计算常数项，用于GELU激活函数的计算公式中
    float s = sqrtf(2.0f / M_PI);
    // 遍历所有输入元素，并计算其GELU激活值
    for (int i = 0; i < N; i++)
    {
        float x = inp[i];
        // 计算x的三次方，并乘以一个固定的系数，这是GELU公式的一部分
        float cube = 0.044715f * x * x * x;
        // 根据GELU的计算公式，计算激活值，并保存到输出数组中
        out[i] = 0.5f * x * (1.0f + tanhf(s * (x + cube)));
    }
}

/**
 * GELU（Gaussian Error Linear Unit）激活函数的反向传播计算。
 * 对输入数据进行GELU激活函数的反向传播，计算输出梯度对输入的导数，并更新输入梯度。
 *
 * @param dinp 输入梯度的输出数组，大小为N。函数执行后，这里存储了GELU激活函数的输入关于输出梯度的导数。
 * @param inp 激活函数的输入数组，大小为N。仅用于计算，不作修改。
 * @param dout 输出梯度的输入数组，大小为N。表示前向传播输出关于最终目标的梯度。
 * @param N 数组的大小，即输入、输出数组中的元素个数。
 */
void gelu_backward(float *dinp, float *inp, float *dout, int N)
{
    // 计算tanh函数参数的预加重系数
    float s = sqrtf(2.0f / M_PI);
    for (int i = 0; i < N; i++)
    {
        // 当前输入元素
        float x = inp[i];
        // 预计算x的三次方项，用于后续计算
        float cube = 0.044715f * x * x * x;
        // 计算tanh函数的参数
        float tanh_arg = s * (x + cube);
        // 计算tanh函数和其倒数sech的值
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        // 计算GELU激活函数的导数
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * s * (1.0f + 3.0f * 0.044715f * x * x);
        // 更新输入梯度，考虑当前输出梯度的影响
        dinp[i] += local_grad * dout[i];
    }
}

/**
 * @brief 计算输入信号的残差
 *
 * 该函数将两个输入信号相加，并将结果存储在一个新的输出信号中。
 *
 * @param out 指向输出信号的指针。该信号存储输入信号的残差。
 * @param inp1 指向第一个输入信号的指针。
 * @param inp2 指向第二个输入信号的指针。
 * @param N 输入信号的长度。
 */
void residual_forward(float *out, float *inp1, float *inp2, int N)
{
    // 遍历信号并计算残差
    for (int i = 0; i < N; i++)
    {
        out[i] = inp1[i] + inp2[i];
    }
}

/**
 * 对两个输入数据数组和一个输出数据数组进行残差反向传播操作
 *
 * @param dinp1 第一个输入数据数组，进行残差更新
 * @param dinp2 第二个输入数据数组，进行残差更新
 * @param dout 输出数据数组，用于更新输入数组的残差
 * @param N 数组的长度
 */
void residual_backward(float *dinp1, float *dinp2, float *dout, int N)
{
    // 遍历数组，对每个元素执行残差更新操作
    for (int i = 0; i < N; i++)
    {
        dinp1[i] += dout[i]; // 对din1数组的每个元素加上输出数组的对应元素
        dinp2[i] += dout[i]; // 对din2数组的每个元素加上输出数组的对应元素
    }
}

/**
 * softmax前向传播函数
 * 用于计算给定未归一化对数概率logits的softmax结果
 *
 * @param probs 输出数组，存储归一化后的概率，尺寸为(B,T,V)
 * @param logits 输入数组，存储未归一化对数概率，尺寸为(B,T,V)
 * @param B 批量大小
 * @param T 时间步数（或序列长度）
 * @param V 词汇量（或类别数）
 */
void softmax_forward(float *probs, float *logits, int B, int T, int V)
{
// 使用OpenMP并行化处理来加速循环
#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            // 计算当前时间步和批次的logits和probs起始位置
            float *logits_bt = logits + b * T * V + t * V;
            float *probs_bt = probs + b * T * V + t * V;

            // 初始化最大logits值和归一化后的概率和
            float maxval = -10000.0f; // 用一个较大的负数暂时代替，之后会更新
            float sum = 0.0f;

            // 计算每个类别的指数并更新最大logits值
            for (int i = 0; i < V; i++)
            {
                if (logits_bt[i] > maxval)
                {
                    maxval = logits_bt[i];
                }
            }

            // 计算每个类别的归一化概率
            for (int i = 0; i < V; i++)
            {
                probs_bt[i] = expf(logits_bt[i] - maxval); // 计算指数部分
                sum += probs_bt[i];                        // 累加概率和
            }

            // 归一化概率
            for (int i = 0; i < V; i++)
            {
                probs_bt[i] /= sum; // 求得归一化概率
            }
        }
    }
}

/**
 * 计算交叉熵损失的前向传播
 *
 * @param losses 用于存储每个位置的损失值的浮点数组，尺寸为(B,T)
 * @param probs 包含概率值的浮点数组，尺寸为(B,T,V)，其中B为批次大小，T为时间步数或序列长度，V为词汇表大小（类数）
 * @param targets 包含目标标签的整数数组，尺寸为(B,T)，其中每个元素指示正确标签在logits中的索引
 * @param B 批次大小
 * @param T 时间步数或序列长度
 * @param V 词汇表大小（类数）
 *
 * 该函数遍历每个样本和时间步，计算对应位置的概率与目标标签的交叉熵损失。
 */
void crossentropy_forward(float *losses,
                          float *probs, int *targets,
                          int B, int T, int V)
{
    // 遍历批次中的每个样本
    for (int b = 0; b < B; b++)
    {
        // 遍历样本中每个时间步
        for (int t = 0; t < T; t++)
        {
            // 计算损失：针对每个位置的目标标签，计算其对应概率的负对数
            float *probs_bt = probs + b * T * V + t * V; // 获取当前样本和时间步的概率数组指针
            int ix = targets[b * T + t];                 // 获取当前样本和时间步的目标标签索引
            losses[b * T + t] = -logf(probs_bt[ix]);     // 计算并存储损失值
        }
    }
}

/**
 * 对输入的logits进行softmax和交叉熵损失的反向传播。
 *
 * @param dlogits 指向需要进行反向传播的logits数组的指针。
 * @param dlosses 指向存储损失梯度的数组的指针。
 * @param probs 指向存储softmax概率的数组的指针。
 * @param targets 指向存储目标标签的整型数组的指针。
 * @param B 批量大小。
 * @param T 时间步（或序列长度）。
 * @param V 候选类别数量。
 */
void crossentropy_softmax_backward(float *dlogits,
                                   float *dlosses, float *probs, int *targets,
                                   int B, int T, int V)
{
    // 针对每个样本和时间步进行反向传播
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float *dlogits_bt = dlogits + b * T * V + t * V; // 计算当前位置的梯度指针
            float *probs_bt = probs + b * T * V + t * V;     // 计算当前位置的概率指针
            float dloss = dlosses[b * T + t];                // 获取当前样本和时间步的损失梯度
            int ix = targets[b * T + t];                     // 获取当前样本和时间步的目标标签
            // 针对所有类别进行反向传播
            for (int i = 0; i < V; i++)
            {
                float p = probs_bt[i];                    // 获取当前类别的概率
                float indicator = i == ix ? 1.0f : 0.0f;  // 判断当前类别是否为目标类别
                dlogits_bt[i] += (p - indicator) * dloss; // 计算logits的梯度
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

#define NUM_PARAMETER_TENSORS 16
// 定义模型参数的结构体
// 包含了多个不同作用的浮点数数组，用于存储模型的不同参数
typedef struct
{
    float *wte;      // 用于编码器-解码器注意力的权重矩阵 (V, C)
    float *wpe;      // 位置编码权重矩阵 (maxT, C)
    float *ln1w;     // 第一层层归一化权重矩阵 (L, C)
    float *ln1b;     // 第一层层归一化偏置 (L, C)
    float *qkvw;     // 注意力查询-键权重矩阵 (L, 3*C, C)
    float *qkvb;     // 注意力查询-键偏置 (L, 3*C)
    float *attprojw; // 注意力投影权重矩阵 (L, C, C)
    float *attprojb; // 注意力投影偏置 (L, C)
    float *ln2w;     // 第二层层归一化权重矩阵 (L, C)
    float *ln2b;     // 第二层层归一化偏置 (L, C)
    float *fcw;      // 全连接层权重矩阵 (L, 4*C, C)
    float *fcb;      // 全连接层偏置 (L, 4*C)
    float *fcprojw;  // 全连接投影权重矩阵 (L, C, 4*C)
    float *fcprojb;  // 全连接投影偏置 (L, C)
    float *lnfw;     // 最后一层层归一化权重矩阵 (C)
    float *lnfb;     // 最后一层层归一化偏置 (C)
} ParameterTensors;

/**
 * 为参数分配内存并指向正确的位置
 *
 * @param params 指向参数张量结构体的指针，该结构体将包含指向各个参数的指针。
 * @param param_sizes 参数大小的数组，每个元素对应一个参数的大小。
 * @return 返回一个指向分配的内存块的指针，该内存块包含了所有参数。
 */
float *malloc_and_point_parameters(ParameterTensors *params, size_t *param_sizes)
{
    size_t num_parameters = 0; // 计算总参数数
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        num_parameters += param_sizes[i];
    }
    // 一次性分配所有参数的内存
    float *params_memory = (float *)malloc(num_parameters * sizeof(float));
    // 定义指针数组，用于指向各个参数张量
    float **ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb};
    float *params_memory_iterator = params_memory; // 内存迭代器，用于遍历参数内存块
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        *(ptrs[i]) = params_memory_iterator;      // 将每个参数张量指向相应位置
        params_memory_iterator += param_sizes[i]; // 更新内存迭代器
    }
    return params_memory;
}

// 定义激活张量结构体，用于存储Transformer模型中的各种中间计算结果
#define NUM_ACTIVATION_TENSORS 23
typedef struct
{
    float *encoded;   // 编码后的输入数据 (B, T, C)
    float *ln1;       // 第一层归一化结果 (L, B, T, C)
    float *ln1_mean;  // 第一层归一化后的均值 (L, B, T)
    float *ln1_rstd;  // 第一层归一化后的标准差 (L, B, T)
    float *qkv;       // QKV矩阵乘法的结果 (L, B, T, 3*C)
    float *atty;      // 注意力分数的温度缩放结果 (L, B, T, C)
    float *preatt;    // 注意力前缀 (L, B, NH, T, T)
    float *att;       // 计算后的注意力权重 (L, B, NH, T, T)
    float *attproj;   // 注意力权重投影结果 (L, B, T, C)
    float *residual2; // 第二层残差连接的结果 (L, B, T, C)
    float *ln2;       // 第二层归一化结果 (L, B, T, C)
    float *ln2_mean;  // 第二层归一化后的均值 (L, B, T)
    float *ln2_rstd;  // 第二层归一化后的标准差 (L, B, T)
    float *fch;       // FC层的输入 (L, B, T, 4*C)
    float *fch_gelu;  // FC层经过GELU激活函数的结果 (L, B, T, 4*C)
    float *fcproj;    // FC层的投影结果 (L, B, T, C)
    float *residual3; // 第三层残差连接的结果 (L, B, T, C)
    float *lnf;       // 最后一层归一化结果 (B, T, C)
    float *lnf_mean;  // 最后一层归一化后的均值 (B, T)
    float *lnf_rstd;  // 最后一层归一化后的标准差 (B, T)
    float *logits;    // 分类或回归的logits结果 (B, T, V)
    float *probs;     // 预测概率 (B, T, V)
    float *losses;    // 训练损失 (B, T)
} ActivationTensors;

/**
 * 动态分配内存并为激活张量指针赋值
 *
 * 本函数负责根据给定的激活张量尺寸，动态分配一片连续的内存，并将这块内存的不同部分指向
 * 活跃张量结构体中的各个成员。它简化了内存管理，确保了激活张量数据的高效存储和访问。
 *
 * @param acts 指向 ActivationTensors 结构体的指针，该结构体包含多个激活张量的指针。
 * @param act_sizes 指向一个 size_t 类型数组的指针，该数组包含了每个激活张量的大小。
 * @return 返回一个指向分配的内存块的指针。该内存块用于存储所有激活张量的数据。
 */
float *malloc_and_point_activations(ActivationTensors *acts, size_t *act_sizes)
{
    // 计算所有激活张量需要的总内存大小
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
    {
        num_activations += act_sizes[i];
    }
    // 分配足够存储所有激活张量数据的内存
    float *acts_memory = (float *)malloc(num_activations * sizeof(float));

    // 定义一个指针数组，用于快速访问 ActivationTensors 结构体中的各个激活张量指针
    float **ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses};

    // 使用迭代器遍历所有激活张量，为它们分别分配内存并更新指针
    float *acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
    {
        *(ptrs[i]) = acts_memory_iterator;    // 更新激活张量指针，指向分配的内存
        acts_memory_iterator += act_sizes[i]; // 移动内存迭代器，为下一个激活张量准备空间
    }
    return acts_memory;
}

/**
 * GPT-2模型配置结构体
 * 用于存储GPT-2模型的各个参数，包括序列最大长度、词汇表大小、层数、注意力头数和通道数等。
 */
typedef struct
{
    int max_seq_len; // 最大序列长度，例如1024
    int vocab_size;  // 词汇表大小，例如50257
    int num_layers;  // 层数，例如12
    int num_heads;   // 注意力机制中的头数，例如12
    int channels;    // 通道数，例如768
} GPT2Config;

// GPT2 结构体定义
// 用于存储GPT-2模型的配置、参数、梯度、激活状态等
typedef struct
{
    GPT2Config config;                         // 模型配置
    ParameterTensors params;                   // 模型参数及其大小
    size_t param_sizes[NUM_PARAMETER_TENSORS]; // 参数尺寸数组
    float *params_memory;                      // 参数内存指针
    int num_parameters;                        // 参数数量

    ParameterTensors grads; // 参数梯度
    float *grads_memory;    // 梯度内存指针

    // AdamW优化器需要的缓冲区
    float *m_memory; // 第一阶矩缓冲区
    float *v_memory; // 第二阶矩缓冲区

    ActivationTensors acts;                   // 模型激活状态及其大小
    size_t act_sizes[NUM_ACTIVATION_TENSORS]; // 激活状态尺寸数组
    float *acts_memory;                       // 激活状态内存指针
    int num_activations;                      // 激活状态数量

    ActivationTensors grads_acts; // 激活状态梯度
    float *grads_acts_memory;     // 激活状态梯度内存指针

    // 其他运行时状态配置
    int batch_size;  // 当前前向传播的批次大小(B)
    int seq_len;     // 当前前向传播的序列长度(T)
    int *inputs;     // 当前前向传播的输入令牌
    int *targets;    // 当前前向传播的目标令牌
    float mean_loss; // 在具有目标的前向传播后，将填充平均损失
} GPT2;

/**
 * 从检查点文件构建GPT-2模型
 *
 * @param model 指向GPT2结构体的指针，用于填充模型参数和配置
 * @param checkpoint_path 检查点文件的路径
 * 该函数不返回任何值。
 */
void gpt2_build_from_checkpoint(GPT2 *model, char *checkpoint_path)
{

    // 从检查点文件读取模型
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (model_file == NULL)
    {
        printf("Error opening model file\n");
        exit(1);
    }
    int model_header[256];
    fread(model_header, sizeof(int), 256, model_file);
    // 检查模型文件的正确性
    if (model_header[0] != 20240326)
    {
        printf("Bad magic model file");
        exit(1);
    }
    if (model_header[1] != 1)
    {
        printf("Bad version in model file");
        exit(1);
    }

    // 读取超参数
    int maxT, V, L, NH, C;
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    printf("[GPT-2]\n");
    printf("max_seq_len: %d\n", maxT);
    printf("vocab_size: %d\n", V);
    printf("num_layers: %d\n", L);
    printf("num_heads: %d\n", NH);
    printf("channels: %d\n", C);

    // 分配参数空间并读取参数
    model->param_sizes[0] = V * C;
    model->param_sizes[1] = maxT * C;
    model->param_sizes[2] = L * C;
    model->param_sizes[3] = L * C;
    model->param_sizes[4] = L * (3 * C) * C;
    model->param_sizes[5] = L * (3 * C);
    model->param_sizes[6] = L * C * C;
    model->param_sizes[7] = L * C;
    model->param_sizes[8] = L * C;
    model->param_sizes[9] = L * C;
    model->param_sizes[10] = L * (4 * C) * C;
    model->param_sizes[11] = L * (4 * C);
    model->param_sizes[12] = L * C * (4 * C);
    model->param_sizes[13] = L * C;
    model->param_sizes[14] = C;
    model->param_sizes[15] = C;

    // 计算参数总数
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // 从文件中读取所有参数
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    fread(model->params_memory, sizeof(float), num_parameters, model_file);
    fclose(model_file);

    // 其他初始化
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f将标记无损失
}

/**
 * 对GPT-2模型进行前向传播。
 *
 * @param model GPT-2模型结构体指针。
 * @param inputs 输入序列的整数数组。
 * @param targets 目标序列的整数数组，可以为NULL。
 * @param B 批量大小。
 * @param T 序列长度。
 */
void gpt2_forward(GPT2 *model, int *inputs, int *targets, int B, int T)
{
    // 目标序列是可选的，可以为NULL

    // 确保模型已正确初始化，否则报错退出
    if (model->params_memory == NULL)
    {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // 便利参数
    int V = model->config.vocab_size; // 词汇表大小
    int L = model->config.num_layers; // 层数
    int NH = model->config.num_heads; // 注意力头数
    int C = model->config.channels;   // 通道数

    // 如果需要，分配所有激活函数的空间（在此处懒惰地完成）
    if (model->acts_memory == NULL)
    {
        // 记录当前的B,T值
        model->batch_size = B;
        model->seq_len = T;
        // 分别为各层和各个步骤分配激活函数空间
        model->act_sizes[0] = B * T * C;
        model->act_sizes[1] = L * B * T * C;
        model->act_sizes[2] = L * B * T;
        model->act_sizes[3] = L * B * T;
        model->act_sizes[4] = L * B * T * 3 * C;
        model->act_sizes[5] = L * B * T * C;
        model->act_sizes[6] = L * B * NH * T * T;
        model->act_sizes[7] = L * B * NH * T * T;
        model->act_sizes[8] = L * B * T * C;
        model->act_sizes[9] = L * B * T * C;
        model->act_sizes[10] = L * B * T * C;
        model->act_sizes[11] = L * B * T;
        model->act_sizes[12] = L * B * T;
        model->act_sizes[13] = L * B * T * 4 * C;
        model->act_sizes[14] = L * B * T * 4 * C;
        model->act_sizes[15] = L * B * T * C;
        model->act_sizes[16] = L * B * T * C;
        model->act_sizes[17] = B * T * C;
        model->act_sizes[18] = B * T;
        model->act_sizes[19] = B * T;
        model->act_sizes[20] = B * T * V;
        model->act_sizes[21] = B * T * V;
        model->act_sizes[22] = B * T;
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
        {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // 同时为输入和目标创建内存缓存
        model->inputs = (int *)malloc(B * T * sizeof(int));
        model->targets = (int *)malloc(B * T * sizeof(int)); // 如果从不使用目标，那么这可能未被利用，但它的内存开销很小
    }
    else
    {
        // 验证B,T是否小于等于之前分配的内存
        // 在原则上，我们可以重新分配更大的内存块，但目前我们仅报错退出
        if (B > model->batch_size || T > model->seq_len)
        {
            printf("Error: batch size or sequence length is inadequately large\n");
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(1);
        }
    }

    // 缓存输入/目标
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL)
    {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // 前向传播
    ParameterTensors params = model->params; // 简化参数
    ActivationTensors acts = model->acts;
    float *residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // 编码进入残差[0]
    for (int l = 0; l < L; l++)
    {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

        // 获取当前层的权重指针
        float *l_ln1w = params.ln1w + l * C;
        float *l_ln1b = params.ln1b + l * C;
        float *l_qkvw = params.qkvw + l * 3 * C * C;
        float *l_qkvb = params.qkvb + l * 3 * C;
        float *l_attprojw = params.attprojw + l * C * C;
        float *l_attprojb = params.attprojb + l * C;
        float *l_ln2w = params.ln2w + l * C;
        float *l_ln2b = params.ln2b + l * C;
        float *l_fcw = params.fcw + l * 4 * C * C;
        float *l_fcb = params.fcb + l * 4 * C;
        float *l_fcprojw = params.fcprojw + l * C * 4 * C;
        float *l_fcprojb = params.fcprojb + l * C;

        // 获取当前层的激活函数指针
        float *l_ln1 = acts.ln1 + l * B * T * C;
        float *l_ln1_mean = acts.ln1_mean + l * B * T;
        float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float *l_qkv = acts.qkv + l * B * T * 3 * C;
        float *l_atty = acts.atty + l * B * T * C;
        float *l_preatt = acts.preatt + l * B * NH * T * T;
        float *l_att = acts.att + l * B * NH * T * T;
        float *l_attproj = acts.attproj + l * B * T * C;
        float *l_residual2 = acts.residual2 + l * B * T * C;
        float *l_ln2 = acts.ln2 + l * B * T * C;
        float *l_ln2_mean = acts.ln2_mean + l * B * T;
        float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float *l_fch = acts.fch + l * B * T * 4 * C;
        float *l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
        float *l_fcproj = acts.fcproj + l * B * T * C;
        float *l_residual3 = acts.residual3 + l * B * T * C;

        // 执行前向传播
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B * T * C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
        gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
    }
    residual = acts.residual3 + (L - 1) * B * T * C; // 最后一个残差在residual3中
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
    softmax_forward(acts.probs, acts.logits, B, T, V);

    // 如果有目标序列，也前向传播交叉熵损失函数
    if (targets != NULL)
    {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, V);
        // 为了方便，也计算损失的平均值
        float mean_loss = 0.0f;
        for (int i = 0; i < B * T; i++)
        {
            mean_loss += model->acts.losses[i];
        }
        mean_loss /= B * T;
        model->mean_loss = mean_loss;
    }
    else
    {
        // 如果没有目标序列，就没有损失
        model->mean_loss = -1.0f;
    }
}
/**
 * 为GPT-2模型清零梯度
 * @param model 指向GPT2模型的指针。模型结构体中包含了梯度内存和其他相关参数。
 */
void gpt2_zero_grad(GPT2 *model)
{
    // 如果梯度内存不为空，则清零梯度内存
    if (model->grads_memory != NULL)
    {
        memset(model->grads_memory, 0, model->num_parameters * sizeof(float));
    }
    // 如果激活梯度内存不为空，则清零激活梯度内存
    if (model->grads_acts_memory != NULL)
    {
        memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float));
    }
}

/**
 * 对GPT-2模型进行反向传播
 *
 * @param model 指向GPT-2模型结构体的指针
 *
 * 注意：在调用此函数之前，必须先使用相应的前向传播函数，并且提供了目标数据。
 * 反向传播过程中，会计算模型参数的梯度，并更新激活函数的梯度状态。
 */
void gpt2_backward(GPT2 *model)
{

    // 检查之前是否已经进行了前向传播，并且提供了目标数据
    if (model->mean_loss == -1.0f)
    {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // 如果尚未分配梯度内存，则进行分配
    if (model->grads_memory == NULL)
    {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model); // 初始化梯度为0
    }

    // 简便变量
    int B = model->batch_size;        // 批量大小
    int T = model->seq_len;           // 序列长度
    int V = model->config.vocab_size; // 词汇表大小
    int L = model->config.num_layers; // 层数
    int NH = model->config.num_heads; // 注意力头数
    int C = model->config.channels;   // 通道数

    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // 初始化梯度，以计算平均损失
    float dloss_mean = 1.0f / (B * T);
    for (int i = 0; i < B * T; i++)
    {
        grads_acts.losses[i] = dloss_mean;
    }

    // 开始反向传播：计算交叉熵和softmax的梯度
    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V);
    // 计算矩阵乘法的梯度
    matmul_backward(grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf, params.wte, B, T, C, V);

    // 初始化最后一个层的残差及其梯度
    float *residual = acts.residual3 + (L - 1) * B * T * C;
    float *dresidual = grads_acts.residual3 + (L - 1) * B * T * C;
    // 计算层归一化的梯度
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    // 逐层进行反向传播
    for (int l = L - 1; l >= 0; l--)
    {

        // 获取当前层的残差及其梯度指针
        residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l - 1) * B * T * C;

        // 获取当前层的权重和梯度的指针
        float *l_ln1w = params.ln1w + l * C;
        float *l_qkvw = params.qkvw + l * 3 * C * C;
        float *l_attprojw = params.attprojw + l * C * C;
        float *l_ln2w = params.ln2w + l * C;
        float *l_fcw = params.fcw + l * 4 * C * C;
        float *l_fcprojw = params.fcprojw + l * C * 4 * C;

        float *dl_ln1w = grads.ln1w + l * C;
        float *dl_ln1b = grads.ln1b + l * C;
        float *dl_qkvw = grads.qkvw + l * 3 * C * C;
        float *dl_qkvb = grads.qkvb + l * 3 * C;
        float *dl_attprojw = grads.attprojw + l * C * C;
        float *dl_attprojb = grads.attprojb + l * C;
        float *dl_ln2w = grads.ln2w + l * C;
        float *dl_ln2b = grads.ln2b + l * C;
        float *dl_fcw = grads.fcw + l * 4 * C * C;
        float *dl_fcb = grads.fcb + l * 4 * C;
        float *dl_fcprojw = grads.fcprojw + l * C * 4 * C;
        float *dl_fcprojb = grads.fcprojb + l * C;

        // 以及当前层的激活函数和梯度的指针
        float *l_ln1 = acts.ln1 + l * B * T * C;
        float *l_ln1_mean = acts.ln1_mean + l * B * T;
        float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float *l_qkv = acts.qkv + l * B * T * 3 * C;
        float *l_atty = acts.atty + l * B * T * C;
        float *l_att = acts.att + l * B * NH * T * T;
        float *l_residual2 = acts.residual2 + l * B * T * C;
        float *l_ln2 = acts.ln2 + l * B * T * C;
        float *l_ln2_mean = acts.ln2_mean + l * B * T;
        float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float *l_fch = acts.fch + l * B * T * 4 * C;
        float *l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;

        float *dl_ln1 = grads_acts.ln1 + l * B * T * C;
        float *dl_qkv = grads_acts.qkv + l * B * T * 3 * C;
        float *dl_atty = grads_acts.atty + l * B * T * C;
        float *dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        float *dl_att = grads_acts.att + l * B * NH * T * T;
        float *dl_attproj = grads_acts.attproj + l * B * T * C;
        float *dl_residual2 = grads_acts.residual2 + l * B * T * C;
        float *dl_ln2 = grads_acts.ln2 + l * B * T * C;
        float *dl_fch = grads_acts.fch + l * B * T * 4 * C;
        float *dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4 * C;
        float *dl_fcproj = grads_acts.fcproj + l * B * T * C;
        float *dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // 对当前层进行反向传播
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);
        // 计算矩阵乘法的梯度
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);
        // 计算GELU激活函数的梯度
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
        // 计算矩阵乘法的梯度
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
        // 计算层归一化的梯度
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        // 计算残差的梯度
        residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C);
        // 计算注意力权重的梯度
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        // 计算注意力机制的梯度
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        // 计算矩阵乘法的梯度
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C);
        // 计算层归一化的梯度
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    // 对编码器部分进行反向传播
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

/**
 * 使用AdamW算法更新GPT-2模型的参数。
 *
 * @param model 指向GPT2结构体的指针，包含模型的参数和其他必要信息。
 * @param learning_rate 学习率，控制参数更新的步长。
 * @param beta1 第一阶矩的衰减因子，用于实现动量累积。
 * @param beta2 第二阶矩的衰减因子，用于实现RMSprop。
 * @param eps 用于避免除以零的小值。
 * @param weight_decay 权重衰减系数，相当于L2正则化的强度。
 * @param t 当前迭代步数。
 */
void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t)
{
    // 如果m_memory和v_memory未初始化，则分配内存
    if (model->m_memory == NULL)
    {
        model->m_memory = (float *)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float *)calloc(model->num_parameters, sizeof(float));
    }

    // 遍历所有参数，进行更新
    for (int i = 0; i < model->num_parameters; i++)
    {
        float param = model->params_memory[i];
        float grad = model->grads_memory[i];

        // 更新第一阶矩（动量）
        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // 更新第二阶矩（RMSprop）
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // 对两个矩进行偏差修正
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // 更新参数
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

/**
 * 释放GPT-2模型的内存资源。
 * @param model 指向GPT2结构体的指针，包含了需要被释放的内存资源。
 */
void gpt2_free(GPT2 *model)
{
    // 释放模型参数的内存
    free(model->params_memory);
    // 释放模型梯度的内存
    free(model->grads_memory);
    // 释放用于Adam优化器的m记忆体
    free(model->m_memory);
    // 释放用于Adam优化器的v记忆体
    free(model->v_memory);
    // 释放激活函数的输出内存
    free(model->acts_memory);
    // 释放梯度激活函数的输出内存
    free(model->grads_acts_memory);
    // 释放输入数据的内存
    free(model->inputs);
    // 释放目标数据的内存
    free(model->targets);
}

#ifndef TESTING
// if we are TESTING (see test.c), we'll skip the int main below

// ----------------------------------------------------------------------------
// data loader lite
// returns random batches of data from a file of integers

/*
 * DataLoader 结构体定义
 * 用于管理数据加载器的状态和配置
 */
typedef struct
{
    // 超参数
    int B; // 批量大小
    int T; // 序列长度
    // 输入处理及其状态
    FILE *tokens_file;     // 保存令牌的文件指针
    long file_size;        // 文件大小，单位为字节
    long current_position; // 当前读取位置，单位为字节
    // 输出内存管理
    int *batch;   // 存储批量数据的指针
    int *inputs;  // 存储输入序列的指针
    int *targets; // 存储目标序列的指针
    // 辅助变量
    int num_batches; // 数据集中总共的批次数
} DataLoader;

/**
 * 初始化数据加载器
 *
 * 该函数负责初始化一个数据加载器对象，包括打开指定的文件，设置批处理大小和序列长度，以及分配内存以供后续读取数据使用。
 *
 * @param loader 数据加载器的指针。该加载器将被初始化。
 * @param filename 需要被读取的数据文件的名称。
 * @param B 批处理大小。每次从文件中读取的样本数量。
 * @param T 序列长度。每个样本的长度。
 */
void dataloader_init(DataLoader *loader, char *filename, int B, int T)
{
    loader->B = B;
    loader->T = T;

    // 尝试打开输入文件以读取数据
    loader->tokens_file = fopen(filename, "rb");
    if (loader->tokens_file == NULL)
    {
        printf("Error opening tokens file\n");
        exit(1);
    }

    // 计算文件大小，以确保文件足够大，能够包含至少一个批次的数据
    fseek(loader->tokens_file, 0, SEEK_END);
    loader->file_size = ftell(loader->tokens_file);
    fseek(loader->tokens_file, 0, SEEK_SET);
    if (loader->file_size < (B * T + 1) * sizeof(int))
    {
        printf("Error: file size is too small for the batch size and sequence length\n");
        exit(1);
    }
    loader->current_position = 0; // 初始化读取位置到文件起始处

    // 分配内存以存储批处理数据和目标数据。额外分配一个整数用于目标数据的起始标记。
    loader->batch = (int *)malloc((B * T + 1) * sizeof(int));
    loader->inputs = loader->batch;
    loader->targets = loader->batch + 1; // 目标数据从输入数据的下一个位置开始
    // 计算文件中能够包含的完整批次数量
    loader->num_batches = loader->file_size / (B * T * sizeof(int));
}

/**
 * 重置数据加载器的状态。
 * @param loader 数据加载器的指针。指向一个DataLoader类型的结构体。
 * 该函数将数据加载器的当前位置重置为起始位置。
 */
void dataloader_reset(DataLoader *loader)
{
    loader->current_position = 0; // 将当前位置重置为0
}

/**
 * 加载器获取下一个批次的数据。
 *
 * @param loader 指向DataLoader结构体的指针，包含数据加载器的状态和配置。
 *
 * 此函数将从数据文件中读取指定大小的批次数据。如果当前读取位置到达文件末尾，
 * 则会循环回到文件的起始位置继续读取。
 */
void dataloader_next_batch(DataLoader *loader)
{
    int B = loader->B; // 批次大小
    int T = loader->T; // 时间步长

    // 检查是否到达文件末尾，如果是，则回到文件起始位置
    if (loader->current_position + (B * T + 1) * sizeof(int) > loader->file_size)
    {
        loader->current_position = 0;
    }

    // 从文件中读取批次数据
    fseek(loader->tokens_file, loader->current_position, SEEK_SET);
    fread(loader->batch, sizeof(int), B * T + 1, loader->tokens_file);

    // 更新当前读取位置，为读取下一个批次做准备
    loader->current_position += B * T * sizeof(int);
}

/**
 * 释放数据加载器资源
 * 该函数负责关闭数据加载器中打开的文件句柄，并释放分配给批次数据的内存。
 *
 * @param loader 数据加载器的指针。这是一个指向 DataLoader 结构体的指针，该结构体包含了数据加载器的相关配置和状态，例如文件句柄和批次数据。
 */
void dataloader_free(DataLoader *loader)
{
    // 关闭数据文件句柄
    fclose(loader->tokens_file);
    // 释放批次数据内存
    free(loader->batch);
}

// ----------------------------------------------------------------------------
// sampler
// 定义GPT2的结束符
#define GPT2_EOT 50256

/**
 * 使用xorshift算法生成一个32位的随机数。
 *
 * @param state 指向一个64位无符号长整型的指针，作为随机数生成的状态。
 * @return 返回一个32位的无符号整型随机数。
 */
unsigned int random_u32(unsigned long long *state)
{
    // xorshift算法的变种，通过位运算更新状态
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    // 使用状态生成并返回随机数
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

/**
 * 生成一个0到1之间的随机浮点数。
 *
 * @param state 指向一个64位无符号长整型的指针，作为随机数生成的状态。
 * @return 返回一个介于0（包含）和1（不包含）之间的随机浮点数。
 */
float random_f32(unsigned long long *state)
{
    // 通过生成的32位随机数修整得到[0,1)内的随机浮点数
    return (random_u32(state) >> 8) / 16777216.0f;
}

/**
 * 根据给定的概率数组，进行概率抽样得到一个索引。
 *
 * @param probabilities 指向包含n个元素的概率数组的指针。
 * @param n 概率数组的大小。
 * @param coin 一个介于0（包含）和1（不包含）之间的随机数，用于抽样。
 * @return 返回抽样得到的索引。
 */
int sample_mult(float *probabilities, int n, float coin)
{
    // 计算累积概率分布函数
    float cdf = 0.0f;
    for (int i = 0; i < n; i++)
    {
        cdf += probabilities[i];
        // 如果coin小于当前的累积概率，则返回当前索引
        if (coin < cdf)
        {
            return i;
        }
    }
    // 在概率总和由于浮点数精度问题不严格等于1时，返回最后一个元素的索引
    return n - 1;
}

// ----------------------------------------------------------------------------
// main training loop
/*
主函数：训练和测试GPT-2模型。
*/

int main()
{

    // 从检查点加载GPT-2模型
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // 根据文件存在情况，构建DataLoader。优先使用tiny_shakespeare训练和验证集。
    char *tiny_stories_train = "data/TinyStories_train.bin";
    char *tiny_stories_val = "data/TinyStories_val.bin";
    char *tiny_shakespeare_train = "data/tiny_shakespeare_train.bin";
    char *tiny_shakespeare_val = "data/tiny_shakespeare_val.bin";
    char *train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    char *val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    int B = 4;  // 批量大小
    int T = 64; // 序列长度
    DataLoader train_loader;
    dataloader_init(&train_loader, train_tokens, B, T);                  // 初始化训练数据加载器
    printf("train dataset num_batches: %d\n", train_loader.num_batches); // 打印训练集批次数
    DataLoader val_loader;
    dataloader_init(&val_loader, val_tokens, B, T);                  // 初始化验证数据加载器
    printf("val dataset num_batches: %d\n", val_loader.num_batches); // 打印验证集批次数
    int val_num_batches = 10;                                        // 验证集评估的批次数

    // 分配内存，用于从模型生成样本
    unsigned long long rng_state = 1337; // 随机数生成器状态
    const int gen_max_length = 64;       // 生成序列的最大长度
    int gen_tokens[gen_max_length];      // 存储生成的令牌

    // 训练过程
    struct timespec start, end;
    for (int step = 0; step <= 40; step++)
    {

        // 每隔一定步数估计验证集损失
        if (step % 10 == 0)
        {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader); // 重置验证数据加载器
            for (int i = 0; i < val_num_batches; i++)
            {
                dataloader_next_batch(&val_loader);                                // 获取下一个验证集批次
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T); // 前向传播
                val_loss += model.mean_loss;                                       // 累加损失
            }
            val_loss /= val_num_batches;       // 计算平均损失
            printf("val loss %f\n", val_loss); // 打印验证集损失
        }

        // 每隔一定步数进行模型推断，打印生成的文本
        if (step > 0 && step % 20 == 0)
        {
            gen_tokens[0] = GPT2_EOT; // 用GPT-2的EOT令牌开始生成
            for (int t = 1; t < gen_max_length; t++)
            {
                // 注意：这里的推断比较浪费，因为对于每个t，我们都要重新计算0到t之间的所有激活值
                // 由于此处的推断只是为了检查模型，因此没有对此进行优化
                gpt2_forward(&model, gen_tokens, NULL, 1, t);                        // 前向传播
                float *probs = model.acts.probs + (t - 1) * model.config.vocab_size; // 获取概率分布
                float coin = random_f32(&rng_state);                                 // 生成随机数
                int next_token = sample_mult(probs, model.config.vocab_size, coin);  // 根据概率分布采样下一个令牌
                gen_tokens[t] = next_token;                                          // 更新生成的令牌序列
            }
            printf("generated: ");
            for (int t = 0; t < gen_max_length; t++)
            {
                printf("%d ", gen_tokens[t]);
            }
            printf("\n");
        }

        // 进行一个训练步骤
        clock_gettime(CLOCK_MONOTONIC, &start);                                                        // 记录开始时间
        dataloader_next_batch(&train_loader);                                                          // 获取下一个训练批次
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);                         // 前向传播
        gpt2_zero_grad(&model);                                                                        // 清零梯度
        gpt2_backward(&model);                                                                         // 反向传播
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);                               // 更新模型参数
        clock_gettime(CLOCK_MONOTONIC, &end);                                                          // 记录结束时间
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;     // 计算耗时
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000); // 打印训练损失和耗时
    }

    // 释放资源
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    gpt2_free(&model);
    return 0;
}
#endif