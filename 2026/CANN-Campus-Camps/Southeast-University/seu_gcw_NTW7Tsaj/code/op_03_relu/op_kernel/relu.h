#ifndef RELU_H
#define RELU_H


#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "relu_tiling_data.h"
#include "relu_tiling_key.h"



namespace NsRelu {


using namespace AscendC;



constexpr int32_t BUFFER_NUM=2;



template<typename T>
class Relu
{

public:


__aicore__ inline Relu(){}


__aicore__ inline void Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData);



__aicore__ inline void Process();



private:


__aicore__ inline void CopyIn(
    int64_t offset,
    int64_t len);



__aicore__ inline void Compute(
    int64_t len);



__aicore__ inline void CopyOut(
    int64_t offset,
    int64_t len);



private:


TPipe pipe;



TQue<
QuePosition::VECIN,
BUFFER_NUM>
inputQueueX;



TQue<
QuePosition::VECOUT,
BUFFER_NUM>
outputQueueY;



GlobalTensor<T> inputGMX;

GlobalTensor<T> outputGMY;



int64_t blockLength_=0;

int64_t ubLength_=0;

int64_t currentLength_=0;


};



template<typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{


    blockLength_ =
        tilingData->blockFactor;



    ubLength_ =
        tilingData->ubFactor;



    int64_t blockIdx =
        GetBlockIdx();



    int64_t start =
        blockIdx *
        blockLength_;



    int64_t remain =
        tilingData->totalNum
        -
        start;



    if(remain<=0)
    {
        currentLength_=0;
        return;
    }



    currentLength_ =
        remain < blockLength_
        ?
        remain
        :
        blockLength_;




    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x+start,
        currentLength_);



    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y+start,
        currentLength_);




    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_*sizeof(T));



    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_*sizeof(T));



}



template<typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t offset,
    int64_t len)
{


    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();



    DataCopy(
        xLocal,
        inputGMX[offset],
        len);



    inputQueueX.EnQue(
        xLocal);

}



template<typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t len)
{


    auto xLocal =
        inputQueueX.DeQue<T>();


    auto yLocal =
        outputQueueY.AllocTensor<T>();


    AscendC::Relu(
        yLocal,
        xLocal,
        len);



    outputQueueY.EnQue(
        yLocal);



    inputQueueX.FreeTensor(
        xLocal);

}



template<typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t offset,
    int64_t len)
{


    auto yLocal =
        outputQueueY.DeQue<T>();



    DataCopy(
        outputGMY[offset],
        yLocal,
        len);



    outputQueueY.FreeTensor(
        yLocal);

}



template<typename T>
__aicore__ inline void Relu<T>::Process()
{


    if(currentLength_<=0)
    {
        return;
    }



    int64_t loop =
        (currentLength_
        +
        ubLength_
        -
        1)
        /
        ubLength_;



    for(int64_t i=0;i<loop;i++)
    {


        int64_t offset =
            i*ubLength_;



        int64_t len =
            (offset+ubLength_
            >
            currentLength_)
            ?
            currentLength_-offset
            :
            ubLength_;



        CopyIn(
            offset,
            len);



        Compute(
            len);



        CopyOut(
            offset,
            len);

    }

}



}
#endif