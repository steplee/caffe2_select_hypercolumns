
#include "hyper_op.h"

/*
 * To understand some of this better, see the caffe2 tut, then some operators in caffe2 source repo, then here:
 *    https://github.com/caffe2/caffe2/blob/0dd3284525079f3870df92f61bed3b94eb45ff53/caffe2/core/operator_schema.h
 */

namespace caffe2 {

  OPERATOR_SCHEMA(Hyper)
    .NumInputs(2, INT_MAX)
    .NumOutputs(1)
    .SetDoc(R"DOC(
        Extracts activations of the given layers at the given pixel locations
        )DOC")

    //.Arg("img_w", "Image Width")
    //.Arg("img_h", "Image Height")

    // Need to find out how to grab blobs from Workspace
    .Input(0, "pixel_location", "Where to trace from. Must be less than w/h of img input size.")
    .Input(1, "sources", "Variable number blobs to draw features from")
    .Output(0, "hyper_columns", "Gathered features of blobs ");

    REGISTER_CPU_OPERATOR(Hyper, HyperOp<CPUContext>);
    //REGISTER_CPU_OPERATOR(HyperGradient, HyperGradientOp<float, CPUContext>);

}
