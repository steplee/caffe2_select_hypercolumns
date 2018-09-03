
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

            // Input(i) returns a const reference to a TensorCPU/CUDA (actually the template type of the op), Output(i) returns a pointer to it.
            // Inputs() returns a const vector to const blob pointers.
            const auto& pixelLocation_t = Input(0);
            const float* pixelLocation = static_cast<float*>(pixelLocation_t.raw_mutable_data());

            const int batchSize = pixelLocation_t.dims()[0];
            const int numLocations = pixelLocation_t.dims()[1];
            CAFFE_ENFORCE(pixelLocation_t.dims()[2] == 2); // x & y

            std::cout << " you are asking " << numLocations << " per batch item. " << std::endl;

            std::cout << " pixelLocation[0][0] as float: " << pixelLocation[0] << " " << pixelLocation[1] << std::endl;

            // Examine each source blob to see how many bytes to allocate for output.
            int32_t out_len = 0;
            TypeMeta src_meta;
            for(int i=1; i<InputSize(); i++) {
                const auto& src = Input(i); // Could also use .InputTensorShapes()

                assert(src.dims().size() == 4); // Make sure it is conv-like activation (rank-4).
                out_len += src.dims()[1] * sizeof(float); // Accumulate channel dim.

                src_meta = src.meta(); // overwritten each loop, should all be same anyways.

                // TODO ensure all types are float32
            }

            std::cout << " Total bytes allocated: " << out_len << std::endl;

            auto* output = Output(0);
            vector<TIndex> outShape = { batchSize, numLocations, out_len };
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

                for (int bi=0; bi<batchSize; bi++) {

                    for (int li=0; li<numLocations; li++) {
                        float relY = pixelLocation[bi * numLocations * 2 + li * 2],
                              relX = pixelLocation[bi * numLocations * 2 + li * 2 + 1];

                        TIndex readY = (int)(actH * relY + .5f);
                        TIndex readX = (int)(actW * relX + .5f);

                        // Standard indexing into a 3D array. Batch is taken care of by batchSrcOff.
                        auto srcOff = (readX * actW * actC
                                    +  readY * actC) * sizeof(float);
                        // (???) Why does it need readX*actW instead of readY*actH? Shouldn't it store by columns of pixels?

                        std::cout << " From src " << i << " CHW " << actC << " " << actH << " " << actW << std::endl;
                        std::cout << "       Reading " << readY << " " << readX << std::endl;

                        context_.CopyItemsSameDevice(
                              src.meta(),
                              actC * sizeof(float),
                              rawSrc + srcOff + batchSrcOff,
                              outRaw + outOff);

                        outOff += actC * sizeof(float);
                    }

                    batchSrcOff += actC * actH * actW * sizeof(float);
                    // outOff has fully advanced to next batch item by accumulating in the numLocations loop.
                }
            }
          }



        private:
          //int32_t img_w, img_h;
    };

}
