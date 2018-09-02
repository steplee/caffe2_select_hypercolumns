#include <iostream>
#include <caffe2/core/common_omp.h>
#include <caffe2/core/context.h>
#include <caffe2/core/logging.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/workspace.h>
#include <caffe2/core/types.h>
#include <caffe2/core/init.h>
#include <caffe2/core/tensor.h>


#include "hyper_op.h"

using namespace std;
using namespace caffe2;

template <class T>
T prod(vector<T>& v) {
    T acc = 1;
    for (t : v) acc *= t;
    return acc;
}

int main(int argc, char** argv) {

    GlobalInit(&argc, &argv);


    Workspace wrk;
    CPUContext *cctx = new CPUContext();

    NetDef initNet, predNet;

    vector<TIndex> fake_size = {1, 64, 200, 80};

    // Simple init net for testing
    {
        auto op = initNet.add_op();
        op->set_type("GivenTensorFill");
        op->add_output("pixel_location");
        auto val = op->add_arg(), shape = op->add_arg();
        shape->set_name("shape");
        shape->add_ints(1);
        shape->add_ints(2);
        val->set_name("values");
        val->add_floats(.5f);
        val->add_floats(.5f);

        op = initNet.add_op();
        op->set_type("GivenTensorFill");
        op->add_output("act0");
        val = op->add_arg(), shape = op->add_arg();
        shape->set_name("shape");
        shape->add_ints(fake_size[0]);
        shape->add_ints(fake_size[1]);
        shape->add_ints(fake_size[2]);
        shape->add_ints(fake_size[3]);
        val->set_name("values");

        int total_eles = prod(fake_size);
        for (int i=0;i<total_eles; i++)
            //val->add_floats(((float)i) / total_eles);
            val->add_floats(i % fake_size[1]); // Correct answer should be 0-#channels.
    }

    // Our operator.
    {
        auto op = predNet.add_op();
        op->set_type("Hyper");
        op->add_input("pixel_location");
        op->add_input("act0");

        auto imgW = op->add_arg(), imgH = op->add_arg();
        imgW->set_name("img_w");
        imgH->set_name("img_h");
        imgW->set_i(100);
        imgH->set_i(100);

        op->add_output("my_hyper_columns");
    }

    initNet.set_name("test_net_init");
    predNet.set_name("test_net");

    wrk.RunNetOnce(initNet);
    wrk.CreateNet(predNet);

    // Test Running HyperOp.
    {
        wrk.RunNet(predNet.name());

        TensorCPU* ans_b = wrk.GetBlob("my_hyper_columns")->GetMutable<TensorCPU>();
        float* ans_t = static_cast<float*>(ans_b->raw_mutable_data());

        std::cout << "Ans:\n";
        for(int i=0; i<fake_size[1]; i++)
            std::cout << " " << ans_t[i];
        std::cout << std::endl;
    }



    std::cout << " -- Tests completed. " << std::endl;

    return 0;
}
