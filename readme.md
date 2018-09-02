# Hyper Column Operator
Stack all features/activations from a single cell of specified blobs, where the chosen cell is the one nearest to a pixel. This could be done as a messy collection of operators setup in a NetDef or even, most simply, accessing the blobs in the workspace, but I wanted to gain the experience of writing a caffe2 op and this seems a good task.

## Usage
 - You need at least two inputs. 
    - First is a batch (TODO, rn only a single point and single batch) of lists of floating-point pixel locations. That is, a 3D tensor of shape BL2 (B for batch, L for list length, 2 for x & y). Note: x comes first.
    - Second comes more than 1 blobs to perform the operation on.

## Notes
  - Caffe2 lazy allocates tensors, you should set up the wanted size then call `raw_mutable_data` or `mutable_data` to create a new one.
  - I'm still a little unclear of tensors vs blobs. Maybe blob is used just for registering with the workspace?
  - The C++ workflow for creating new ops is exactly like in Tensorflow.
  - The documentation is pretty lacking, but the source code is easier to navigate then other frameworks.

### References
 1. https://caffe2.ai/docs/custom-operators.html
 2. https://github.com/leonardvandriel/caffe2_cpp_tutorial
