/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"
#include "less_equal_tiling.h"
#include <type_traits>

using T = int8_t;
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queuea
constexpr float NEGATIVE_ONE_FP32 = -1.0F;
constexpr float POSITIVE_ONE_FP32 = 1.0F;
constexpr int32_t NEGATIVE_ONE_I32 = -1;
constexpr int32_t POSITIVE_ONE_I32 = 1;
constexpr float MIN_ACCURACY_FP16 = 0.00000005960464477539063F;
constexpr float MAX_MUL_FP16 = 4096;
constexpr float MIN_ACCURACY_FP32 = 1.1754943508222875e-38;
constexpr float MAX_MUL_1_FP32 = 1125899906842624;
constexpr float MAX_MUL_2_FP32 = 67108864;

template <typename typeT>
__aicore__ inline void DataCopyPadCustom_GM2UB(
    const AscendC::LocalTensor<typeT>& dstLocal, const AscendC::GlobalTensor<typeT>& srcGlobal,
    const uint32_t calCount) {
    if (calCount < BLOCK_SIZE / sizeof(typeT)) {  // 少于32B的数据直接赋值
        for (uint32_t i = 0; i < calCount; i++) {
            dstLocal.SetValue(i, srcGlobal.GetValue(i));
        }
    }
    else {  // 多于32B的数据先将32B的倍数copy，剩下不对齐的再赋值
        uint32_t padDataCount = calCount - (calCount % (BLOCK_SIZE / sizeof(typeT)));
        AscendC::DataCopy(dstLocal, srcGlobal, padDataCount);
        for (uint32_t i = 0; i < (calCount % (BLOCK_SIZE / sizeof(typeT))); i++) {
            dstLocal[padDataCount].SetValue(i, srcGlobal[padDataCount].GetValue(i));
        }
    }
}

template <typename typeT>
__aicore__ inline void DataCopyPadCustom_UB2GM(
    const AscendC::GlobalTensor<typeT>& dstGlobal, const AscendC::LocalTensor<typeT>& srcLocal,
    const uint32_t calCount) {
    if (calCount < BLOCK_SIZE / sizeof(typeT)) {
        for (uint32_t i = 0; i < calCount; i++) {
            typeT localValue = srcLocal.GetValue(i);
            auto cursor = dstGlobal.address_ + i;
            *cursor = localValue;
        }
    }
    else {
        uint32_t padDataCount = calCount - (calCount % (BLOCK_SIZE / sizeof(typeT)));
        AscendC::DataCopy(dstGlobal, srcLocal, padDataCount);
        for (uint32_t i = 0; i < (calCount % (BLOCK_SIZE / sizeof(typeT))); i++) {
            typeT localValue = srcLocal[padDataCount].GetValue(i);
            auto cursor = dstGlobal[padDataCount].address_ + i;
            *cursor = localValue;
        }
    }
}


template<typename typeT>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, SplitTilingData tiling)
    {
        ResovleTiling(tiling);
        x1_gm.SetGlobalBuffer(
            (__gm__ typeT*)x1 + this->block_offset * AscendC::GetBlockIdx(),
            this->block_length);
        x2_gm.SetGlobalBuffer(
            (__gm__ typeT*)x2 + this->block_offset * AscendC::GetBlockIdx(),
            this->block_length);
        y_gm.SetGlobalBuffer((__gm__ int8_t*)y + this->block_offset * AscendC::GetBlockIdx(),
            this->block_length);

        pipe.InitBuffer(x1_inque, BUFFER_NUM, this->tile_cache * sizeof(typeT));
        pipe.InitBuffer(x2_inque, BUFFER_NUM, this->tile_cache * sizeof(typeT));
        pipe.InitBuffer(y_outque, BUFFER_NUM,
            this->tile_cache * sizeof(int8_t) < BLOCK_SIZE
            ? BLOCK_SIZE
            : this->tile_cache * sizeof(int8_t));
        pipe.InitBuffer(calc_buf_1, this->tile_cache * sizeof(typeT));
        pipe.InitBuffer(calc_buf_2, this->tile_cache * sizeof(half) < BLOCK_SIZE
            ? BLOCK_SIZE
            : this->tile_cache * sizeof(half));
        pipe.InitBuffer(calc_buf_3, this->tile_cache * sizeof(half) < BLOCK_SIZE
            ? BLOCK_SIZE
            : this->tile_cache * sizeof(half));
        pipe.InitBuffer(calc_buf_4, this->tile_cache * sizeof(float) < BLOCK_SIZE
            ? BLOCK_SIZE
            : this->tile_cache * sizeof(float));
    }
    __aicore__ inline void Process() {
        if (this->total_length <= BLOCK_SIZE / sizeof(typeT)) {
            CopyInPad(0);
            Compute(0);
            CopyOutPad(0);
            return;
        }
        int32_t loopCount = this->tile_num;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        if (AscendC::GetBlockIdx() == (AscendC::GetBlockNum() - 1)) {
            CopyInPad(loopCount - 1);
            Compute(loopCount - 1);
            CopyOutPad(loopCount - 1);
        }
        else {
            CopyIn(loopCount - 1);
            Compute(loopCount - 1);
            CopyOut(loopCount - 1);
        }
    }

private:
    __aicore__ inline void ResovleTiling(SplitTilingData tiling)
    {
        uint32_t pad32 = BLOCK_SIZE / sizeof(typeT); // 对齐32B需要的最小数据量
        this->total_length = tiling.totalLength;
        if (AscendC::GetBlockNum() >= 1 && AscendC::GetBlockIdx() == (AscendC::GetBlockNum() - 1)) {
            this->block_length = tiling.blockLengthEnd;
            this->tile_num = tiling.tileNumEnd;
        }
        else {
            this->block_length = tiling.blockLengthMean;
            this->tile_num = tiling.tileNumMean;
        }
        this->block_offset = tiling.blockLengthMean;
        this->tile_length = tiling.tileLengthMean;
        this->tile_cache = tiling.tileLengthMean;
        this->tile_length_end = tiling.tileLengthEnd;
        if (total_length < pad32) {
            this->block_offset = 0;
            this->tile_cache = pad32;
            this->block_length = pad32;
        }
    }
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<typeT> x1_local = x1_inque.AllocTensor<typeT>();
        AscendC::LocalTensor<typeT> x2_local = x2_inque.AllocTensor<typeT>();
        AscendC::DataCopy(x1_local, x1_gm[progress * this->tile_cache], this->tile_cache);
        AscendC::DataCopy(x2_local, x2_gm[progress * this->tile_cache], this->tile_cache);
        x1_inque.EnQue(x1_local);
        x2_inque.EnQue(x2_local);
    }
    __aicore__ inline void CopyInPad(int32_t progress) {
        AscendC::LocalTensor<typeT> x1_local = x1_inque.AllocTensor<typeT>();
        AscendC::LocalTensor<typeT> x2_local = x2_inque.AllocTensor<typeT>();
        DataCopyPadCustom_GM2UB(x1_local, x1_gm[progress * this->tile_cache],
            this->tile_length_end);
        DataCopyPadCustom_GM2UB(x2_local, x2_gm[progress * this->tile_cache],
            this->tile_length_end);
        x1_inque.EnQue(x1_local);
        x2_inque.EnQue(x2_local);
    }
    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<typeT> x1_local = x1_inque.DeQue<typeT>();
        AscendC::LocalTensor<typeT> x2_local = x2_inque.DeQue<typeT>();
        AscendC::LocalTensor<int8_t> y_local = y_outque.AllocTensor<int8_t>();
        AscendC::LocalTensor<typeT> y_compute = calc_buf_1.Get<typeT>();

        if constexpr (std::is_same_v<typeT, half>) {
            AscendC::Max(y_compute, x1_local, x2_local, this->tile_cache);
            AscendC::Sub(y_compute, x2_local, y_compute, this->tile_cache);
            AscendC::Abs(y_compute, y_compute, this->tile_cache);
            AscendC::Mins(y_compute, y_compute, (half)MIN_ACCURACY_FP16, this->tile_cache);
            AscendC::Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->tile_cache);
            AscendC::Muls(y_compute, y_compute, (half)MAX_MUL_FP16, this->tile_cache);
            AscendC::Adds(y_compute, y_compute, (half)NEGATIVE_ONE_FP32, this->tile_cache);
            AscendC::Abs(y_compute, y_compute, this->tile_cache);

            AscendC::Cast(y_local, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
        }
        else if constexpr (std::is_same_v<typeT, float>) {
            AscendC::LocalTensor<half> y_fp16 = calc_buf_2.Get<half>();

            AscendC::Max(y_compute, x1_local, x2_local, this->tile_cache);
            AscendC::Sub(y_compute, x2_local, y_compute, this->tile_cache);
            AscendC::Abs(y_compute, y_compute, this->tile_cache);
            AscendC::Mins(y_compute, y_compute, (float)MIN_ACCURACY_FP32, this->tile_cache);
            AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_1_FP32, this->tile_cache);
            AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_1_FP32, this->tile_cache);
            AscendC::Muls(y_compute, y_compute, (float)MAX_MUL_2_FP32, this->tile_cache);
            AscendC::Adds(y_compute, y_compute, (float)NEGATIVE_ONE_FP32, this->tile_cache);
            AscendC::Abs(y_compute, y_compute, this->tile_cache);

            AscendC::Cast(y_fp16, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
            AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
        }
        else if constexpr (std::is_same_v<typeT, int8_t>) {
            AscendC::LocalTensor<half> x1_local_fp16 = calc_buf_2.Get<half>();
            AscendC::LocalTensor<half> x2_local_fp16 = calc_buf_3.Get<half>();
            AscendC::LocalTensor<half> y_local_fp16 = calc_buf_4.Get<half>();

            AscendC::Cast(x1_local_fp16, x1_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);
            AscendC::Cast(x2_local_fp16, x2_local, AscendC::RoundMode::CAST_NONE, this->tile_cache);

            AscendC::Min(y_local_fp16, x1_local_fp16, x2_local_fp16, this->tile_cache);
            AscendC::Sub(y_local_fp16, x2_local_fp16, y_local_fp16, this->tile_cache);
            AscendC::Mins(y_local_fp16, y_local_fp16, (half)POSITIVE_ONE_FP32,
                this->tile_cache);

            AscendC::Sub(x1_local_fp16, x1_local_fp16, x2_local_fp16, this->tile_cache);
            AscendC::Abs(x1_local_fp16, x1_local_fp16, this->tile_cache);
            AscendC::Mins(x1_local_fp16, x1_local_fp16, (half)POSITIVE_ONE_FP32,
                this->tile_cache);
            AscendC::Duplicate(x2_local_fp16, (half)POSITIVE_ONE_FP32, this->tile_cache);
            AscendC::Sub(x1_local_fp16, x2_local_fp16, x1_local_fp16, this->tile_cache);

            AscendC::Add(y_local_fp16, y_local_fp16, x1_local_fp16, this->tile_cache);

            AscendC::Cast(y_local, y_local_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
        }
        else if constexpr (std::is_same_v<typeT, int32_t>) {
            AscendC::LocalTensor<half> y_fp16 = calc_buf_3.Get<half>();
            AscendC::LocalTensor<float> y_fp32 = calc_buf_4.Get<float>();

            AscendC::Min(y_compute, x1_local, x2_local, this->tile_cache);
            AscendC::Sub(y_compute, x2_local, y_compute, this->tile_cache);
            AscendC::Mins(y_compute, y_compute, (int32_t)POSITIVE_ONE_I32, this->tile_cache);

            AscendC::Sub(x1_local, x1_local, x2_local, this->tile_cache);
            AscendC::Mins(x1_local, x1_local, (int32_t)POSITIVE_ONE_I32, this->tile_cache);
            AscendC::Maxs(x1_local, x1_local, (int32_t)NEGATIVE_ONE_I32, this->tile_cache);
            AscendC::Mul(x1_local, x1_local, x1_local, this->tile_cache);
            AscendC::Duplicate(x2_local, (int32_t)POSITIVE_ONE_I32, this->tile_cache);
            AscendC::Sub(x1_local, x2_local, x1_local, this->tile_cache);

            AscendC::Add(y_compute, y_compute, x1_local, this->tile_cache);

            AscendC::Cast(y_fp32, y_compute, AscendC::RoundMode::CAST_NONE, this->tile_cache);
            AscendC::Cast(y_fp16, y_fp32, AscendC::RoundMode::CAST_NONE, this->tile_cache);
            AscendC::Cast(y_local, y_fp16, AscendC::RoundMode::CAST_NONE, this->tile_cache);
        }

        y_outque.EnQue<int8_t>(y_local);
        x1_inque.FreeTensor(x1_local);
        x2_inque.FreeTensor(x2_local);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<int8_t> y_local = y_outque.DeQue<int8_t>();
        DataCopyPadCustom_UB2GM(y_gm[progress * this->tile_cache], y_local,
            this->tile_cache);
        y_outque.FreeTensor(y_local);
    }
    __aicore__ inline void CopyOutPad(int32_t progress) {
        AscendC::LocalTensor<int8_t> y_local = y_outque.DeQue<int8_t>();
        DataCopyPadCustom_UB2GM(y_gm[progress * this->tile_cache], y_local,
            this->tile_length_end);
        y_outque.FreeTensor(y_local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_1, calc_buf_2, calc_buf_3, calc_buf_4;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x1_inque, x2_inque;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> y_outque;
    AscendC::GlobalTensor<typeT> x1_gm, x2_gm;
    AscendC::GlobalTensor<int8_t> y_gm;
    uint32_t total_length, block_length, block_offset, tile_num, tile_cache,
        tile_length, tile_length_end;
};


extern "C" __global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, SplitTilingData tiling)
{
    KernelLessEqual<T> op;
    op.Init(x1, x2, y, tiling);
    op.Process();
}
