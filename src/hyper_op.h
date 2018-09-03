
#include <cmath>
#include <map>
#include <utility>

#include <caffe2/core/common_omp.h>
#include <caffe2/core/context.h>
#include <caffe2/core/logging.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/types.h>
#include <caffe2/operators/gather_op.h>
#include <caffe2/utils/conversions.h>
#include <caffe2/utils/math.h>


namespace caffe2 {

    template<
        class Context,
        class Engine = DefaultEngine>
    class HyperOp : public Operator<Context> {
        public:
          USE_OPERATOR_CONTEXT_FUNCTIONS;
          HyperOp(const OperatorDef& operator_def, Workspace* ws)
              : Operator<Context>(operator_def, ws)
                //img_w(this->template GetSingleArgument<int32_t>("img_w",0)),
                //img_h(this->template GetSingleArgument<int32_t>("img_h",0))
          {
          }

          ~HyperOp() {}


          // Main op code.
          bool RunOnDevice() override {
            assert(InputSize() >= 2);

            // Input(i) returns a TensorCPU/CUDa
            const auto& pixelLocation_t = Input(0);
            const float* pixelLocation = static_cast<float*>(pixelLocation_t.raw_mutable_data());

            std::cout << " pixelLocation as float: " << pixelLocation[0] << " " << pixelLocation[1] << std::endl;

            // Position of screen we are interested in, in normalized device coords
            // Finding which spatial cell of the activations to read is just multiplying that layer's WH by the respective relXY
            //float relY = pixelLocation[0] / ((float)img_h), relX = pixelLocation[1] / ((float)img_w);
            float relY = pixelLocation[1], relX = pixelLocation[0];

            std::cout << " rel: " << relY << " " << relX << std::endl;


            // Examine each source blob to see how many bytes to allocate for output.
            int32_t out_len = 0;
            int32_t out_batch_size = 0;
            TypeMeta src_meta;
            for(int i=1; i<InputSize(); i++) {
                const auto& src = Input(i);

                assert(src.dims().size() == 4); // Make sure it is conv-like activation (rank-4).
                out_len += src.dims()[1] * sizeof(float); // Accumulate channel dim.

                out_batch_size = src.dims()[0]; // These are overwritten each loop, should all be same anyways.
                src_meta = src.meta();

                // TODO ensure all types are float32
            }

            std::cout << " Total bytes allocated: " << out_len << std::endl;

            auto* output = Output(0);
            vector<TIndex> outShape = { out_batch_size, out_len };
            output->Resize(outShape);

            // Don't forget! Caffe2 lazy-allocates tensors, so you should call Resize, followed by either
            // raw_mutable_data or mutable_data. 
            auto* outRaw = output->raw_mutable_data(src_meta);
            auto outOff = 0;

            for(int i=1; i<InputSize(); i++) {
                const auto& src = Input(i);

                auto batchSrcOff = 0, batchOutOff = 0;

                // we know it is float, but CopyItemsSameDevice needs bytes.
                const uint8_t* rawSrc = static_cast<const uint8_t*>(src.raw_data());

                const auto& dims = src.dims();
                auto batchSize = dims[0], actC = dims[1], actH = dims[2], actW = dims[3];

                TIndex readY = (int)(actH * relY + .5f);
                TIndex readX = (int)(actW * relX + .5f);

                // Standard indexing into a 3D array. Batch is taken care of by batchSrcOff.
                auto srcOff = (readY * actH * actC
                            +  readX * actC) * sizeof(float);

                std::cout << " From src " << i << " CHW " << actC << " " << actH << " " << actW << std::endl;
                std::cout << "       Reading " << readY << " " << readX << std::endl;

                for (int bi=0; bi<batchSize; bi++) {
                    context_.CopyItemsSameDevice(
                          src.meta(),
                          actC * sizeof(float),
                          rawSrc + srcOff + batchSrcOff,
                          outRaw + outOff + batchOutOff);

                    batchSrcOff += actC * actH * actW * sizeof(float);
                    batchOutOff += out_len; // skip to next row.
                }

                outOff += actC * sizeof(float); // advance to next columns.
            }
          }



        private:
          //int32_t img_w, img_h;
    };

}
