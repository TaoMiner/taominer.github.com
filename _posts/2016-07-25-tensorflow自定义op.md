---
layout: post
title:  tensorflow自定义op
date:   2016-07-25 15:14:54
category: coding
tags: [tensorflow,python]
---

* content
{:toc}




看到tensorflow官方教程中word2vec.py中这里不太明白，涉及到自定义op，查了一些资料，中文较少，所以记录一下。

```python
import tensorflow as tf
(words, counts, words_per_epoch, self._epoch, self._words, examples,
     labels) = word2vec.skipgram(filename=opts.train_data,
                                 batch_size=opts.batch_size,
                                 window_size=opts.window_size,
                                 min_count=opts.min_count,
                                 subsample=opts.subsample)
```

查看word2vec.skipgram源码发现是一个自动生成文件gen_word2vec.py:

```python
"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""
def skipgram(filename, batch_size, window_size=None, min_count=None,
             subsample=None, name=None):
 
  return _op_def_lib.apply_op("Skipgram", filename=filename,
                              batch_size=batch_size, window_size=window_size,
                              min_count=min_count, subsample=subsample,
                              name=name)


ops.RegisterShape("Skipgram")(None)
```

而且只定义了一个接口，返回的明显是一个自定义的op，可是到这里就无法再查看具体定义op的实现源码了，因为这里的skipgram使用了静态方式直接编译到tensorflow中去了。后来在官方的[github](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/models/embedding)上找到了对应的.cc文件，是实用c++ kernel实现的自定义op。

除了静态方式，还有一种动态链接.so文件的方式，稍后详细介绍下这种方式的实现。

关于自定义op的两种方式其实在stack overflow的问题[using op registration and kernel linking in tensorflow](http://stackoverflow.com/questions/37548662/understand-op-registration-and-kernel-linking-in-tensorflow/37556646#37556646)介绍的很清楚了。自定义op包含两步：

1. 注册op，主要是定义接口，使用REGISTER_OP()宏；
2. 注册一个或者多个kernel，主要是在不同的kernel上实现op的算法（比如c++），这个即是最终运行的语言，使用REGISTER_KERNEL_BUILDER()宏；

实现之后有两种机制在tensorflow中注册：

1. 静态链接，即把写好的.cc文件放入源码tensorflow/core/user_ops文件夹下，重新编译，然后就可以像上文的skipgram op一样使用了；
2. 动态链接，把写好的.cc文件预先编译成.so文件，然后使用的时候使用tf.load_op_library()函数动态加载即可。

现在详细介绍下动态链接方式的步骤，这里以官方的[zero_out op](https://www.tensorflow.org/versions/r0.9/how_tos/adding_an_op/index.html)为例：

1. 新建zero_out.cc文件，如下：这里将注册op和实现注册kernel写到同一个文件中。

   ```python
   #include "tensorflow/core/framework/op.h"
   #include "tensorflow/core/framework/op_kernel.h"

   using namespace tensorflow;
   using std::vector;
   //@TODO add weight as optional input
   REGISTER_OP("ZeroOut")
       .Input("to_zero: int32")
       .Output("zeroed: int32");

   class ZeroOutOp : public OpKernel {
    public:
     explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}
     void Compute(OpKernelContext* context) override {
       // 获取输入 tensor.
       const Tensor& input_tensor = context->input(0);
       auto input = input_tensor.flat<int32>();
      // 创建一个输出 tensor.
       Tensor* output_tensor = NULL;
       OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                        &output_tensor));
       auto output = output_tensor->template flat<int32>();
       // 设置 tensor 除第一个之外的元素均设为 0.
       const int N = input.size();
       for (int i = 1; i < N; i++) {
         output(i) = 0;
       }
       // 尽可能地保留第一个元素的值.
       if (N > 0) output(0) = input(0);
     }
   };
   REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
   ```

2. 使用c＋＋编译器编译为.so文件，这里使用g＋＋。注意只有max osx需要最后一个-undefined dynamic_lookup。

   ```shell
   TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

   g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $TF_INC -undefined dynamic_lookup
   ```

3. 将编译好的zero_out.so文件放到tf.resource_loader.get_data_files_path()路径下。

4. 接下来我们就可以在程序中通过tf.load_op_library('zero_out.so')函数使用了。运行如下程序验证：

   ```python
   import tensorflow as tf
   _zero_out_module = tf.load_op_library('zero_out.so')
   zero_out = _zero_out_module.zero_out

   with tf.Session(''):
     print zero_out([[1, 2], [3, 4]]).eval()

   '''print'''
   [[1 0]
   [0 0]]
   ```


#### 可能的问题

### no member named 'GetAttr' in 'tensorflow::OpKernelContext'

这是因为tensorflow的op需要用属性进行初始化，因此GetAttr这个方法只能放在构造函数中，如果放在了compute函数就会报错。这也是属性和输入一个不同之处。

3500:100000