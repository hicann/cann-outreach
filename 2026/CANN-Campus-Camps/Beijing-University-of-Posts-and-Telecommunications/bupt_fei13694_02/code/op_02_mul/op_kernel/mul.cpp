// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"
constexpr int BUFFER_NUM=2;
template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        this->tileNum = 8;
        this->blockLength = length / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        xgm.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        ygm.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zgm.SetGlobalBuffer((__gm__ DT_X*)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(in_x, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(in_y, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(out_z, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        int l=this->tileNum*BUFFER_NUM;
        for(int i=0;i<l;i++){
            Copyin(i);
            Compute(i);
            Copyout(i);
        }
    }
private:
    __aicore__ inline void Copyin(int i){
        AscendC::LocalTensor<DT_X> local_x=in_x.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> local_y=in_y.AllocTensor<DT_X>();
        AscendC::DataCopy(local_x,xgm[i*this->tileLength],this->tileLength);
        AscendC::DataCopy(local_y,ygm[i*this->tileLength],this->tileLength);
        in_x.EnQue<DT_X>(local_x);
        in_y.EnQue<DT_X>(local_y);
    }
    __aicore__ inline void Compute(int i){
        AscendC::LocalTensor<DT_X> local_x=in_x.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> local_y=in_y.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> local_z=out_z.AllocTensor<DT_X>();
        AscendC::Mul(local_z,local_x,local_y,this->tileLength);
        out_z.EnQue(local_z);
        in_x.FreeTensor(local_x);
        in_y.FreeTensor(local_y);
    }
    __aicore__ inline void Copyout(int i){
        AscendC::LocalTensor<DT_X> local_z=out_z.DeQue<DT_X>();
        AscendC::DataCopy(zgm[i*tileLength],local_z,this->tileLength);
        out_z.FreeTensor(local_z);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN,BUFFER_NUM> in_x;
    AscendC::TQue<AscendC::QuePosition::VECIN,BUFFER_NUM> in_y;
    AscendC::TQue<AscendC::QuePosition::VECOUT,BUFFER_NUM> out_z;
    AscendC::GlobalTensor<DT_X>xgm;
    AscendC::GlobalTensor<DT_X>ygm;
    AscendC::GlobalTensor<DT_X>zgm;
    int tileNum;
    int tileLength;
    int blockLength;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
