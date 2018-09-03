#include <iostream>
#include <caffe2/core/common_omp.h>
#include <caffe2/core/context.h>
#include <caffe2/core/logging.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/workspace.h>
#include <caffe2/core/types.h>
#include <caffe2/core/init.h>
#include <caffe2/core/tensor.h>
#include <caffe2/core/graph.h>


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

    vector<TIndex> fake_size = {2, 64, 200, 80};

    // Simple init net for testing
    {
        auto op = initNet.add_op();
        op->set_type("GivenTensorFill");
        op->add_output("pixel_location");
        auto val = op->add_arg(), shape = op->add_arg();
        shape->set_name("shape");
        shape->add_ints(2);
        shape->add_ints(1);
        shape->add_ints(2);
        val->set_name("values");
        val->add_floats(.5f);
        val->add_floats(.5f);
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
        for (int i=0;i<total_eles/2; i++)
            val->add_floats(i % fake_size[1]); // Correct answer should be 0-#channels.

        for (int i=0;i<total_eles/2; i++)
            val->add_floats(i % (fake_size[1]/2)); // Correct answer should be 0-#channels/2.
    }

    // Our operator.
    {
        // Better API from core/graph.h
        auto op = AddOp(&predNet, "Hyper", {"pixel_location", "act0"}, {"my_hyper_columns"});

        auto imgW = op->add_arg(), imgH = op->add_arg();
        imgW->set_name("img_w");
        imgH->set_name("img_h");
        imgW->set_i(100);
        imgH->set_i(100);
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
        for (int bi=0; bi<fake_size[0]; bi++) {
            for(int i=0; i<fake_size[1]; i++)
                std::cout << " " << ans_t[bi*fake_size[1] + i];
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }



    std::cout << " -- Tests completed. " << std::endl;

    return 0;
}
