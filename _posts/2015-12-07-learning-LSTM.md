---
layout: post
category: machine learning
title: learning LSTM
---

## 关于LSTM的理解

> 观点均学习自以下文章：
> 
> - Alex Graves大神的博士论文 ["Supervised sequence learning with RNN"](http://www.cs.toronto.edu/~graves/preprint.pdf);
> - LSTM 原始论文 “Long short-term memory”
> - [知乎上推荐的lstm公式推导](http://blog.csdn.net/a635661820/article/details/45390671)

LSTM(long short-term memory)长短期记忆模型是RNN(Recurrent Neural Network)一个变种，以上两种模型又都是人工神经网络模型(ANN)在sequence labelling应用上的改造。传统的ANN通常以固定feature window作为输入（暂不考虑CNN），对比序列作为输入的RNN有几个实际的问题。首先，从语言模型角度来看，context window的长度难以确定，因为文档中对于当前词的有用的上下文的具体位置不是确定的；其次，从语音识别和图像处理角度来看，当输入数据发生失真或者漂移时（同一段语音拉伸成不同的时间播放），那么固定窗口截取的特征变得完全不同，而以序列输入时，则由于前后次序的考虑，对这种变化有较强的鲁棒性。而LSTM又是在此基础上，增强了RNN对有用信息的记忆。这时因为RNN在处理过长的输入序列时，新的信息会覆盖掉旧的信息，从而产生被动遗忘，而LSTM巧妙的利用了三个gate（input，output，forget）：1）主动控制记忆单元中哪些维度可以遗忘，即不是将记忆单元看作整体看待，而是以更小的粒度控制特征的组合（实际上，ANN就是对输入数据特征的重新组合）；2）主动控制哪些维度输入或者输出，跟上一点的原理一样，同时也是为了解决CEC（Constant Error Carrousel），这也是LSTM的核心特征。

了解这种渊源之后，我们也就确定了学习的顺序：ANN—>RNN—>LSTM。那么，首先来直观的看个应用，RNN究竟能做什么。然后依次学习。

[youtube上hinton大牛给出的demo](https://www.youtube.com/watch?v=mLxsbWAYIpw)

### MLP (Multilayer Perceptrons)

ANN网络中没有循环被称为FNN（Feedforword Neural Networks）。下图展示了前馈FNN（左侧）和循环神经网络RNN（右侧）的对比。

![MLP example](/public/img/posts/DPLearning/MLP.png)![RNN](/public/img/posts/DPLearning/RNN.png)



MLP多层感知机是一种被广泛使用的FNN，网络中每一个节点都是一个perceptron。perceptron是一个最简单的人工神经网络模型，节点中使用的函数是线性模型，属于线性的二元分类器，感兴趣的同学可以去[wiki](https://zh.wikipedia.org/wiki/%E6%84%9F%E7%9F%A5%E5%99%A8)了解下。在MLP中，节点中可以使用哪些函数呢？比如常见的有： ![node_function](/Users/ethan/GitHub//public/img/posts/DPLearning/node_function.png)

1. 这个函数应该比较简单，符合单个神经元的特点，同时也没必要复杂，因为一个复杂的神经元又可以转换为若干个简单的神经元；
2. 使用线性还是非线性的呢？这里我们再回头看一下网络图，除了节点中的运算，网络还存在另一种运算，即节点i到j的边上的运算。这个运算是简单的线性乘法: $a_j=\omega_{ij}b_i$，其中**a和b分别代表一个节点的输入和输出**。（牢记这个，否则下面的公式容易混）。因此，若节点中也使用线性函数，则整个网络结构无论怎么调整，依旧是线性函数。这大大限制了MLP的作用。所以使用非线性函数。
3. 经过上面的分析，现在还剩下三个函数，其中sin函数需要分段来确保saturation，不漂亮。所以最常用的有tanh和logistic(sigmoid)函数。而这两个函数是可以相互转换的:

$$
\tanh(x)=2\sigma(2x)-1
$$



因此，MLP几乎可以近似任意的非线性函数(*the universal approximation theory*)；

言归正传，下面正式进入MLP的数学推导。MLP的模型是神经网络，结构上包含input，hidden，output，策略是损失函数$\mathcal{L}$，优化算法是梯度下降。由于模型比较复杂，所以比较有效率的计算梯度下降需要分两步：1）forward pass和2）backpropagation。所以我们的分析也从这几个方面入手，而分析中最强有力的工具是chain rule：

$$
\frac{dz}{dx}=\frac{dz}{dy}\frac{dy}{dx}
$$

只要牢记这一点，并且**明确求导的最终对象**，公式的推导很容易看懂。

#### forward pass

首先来看前向传播，假设MLP中第l隐藏层H中单元h，它的输入$a_h$、输出(激活值)$b_h$、映射(激活)函数$\theta_{h}$。参数节点i到j的权重为$w_{ij}$，则有：

$$
a_h=\mathop{\sum}_{h'\in H_{l-1}}w_{h'h}b_{h'}  \\
b_h=\theta_h(a_h)
$$

#### input layer

输入层可以看作特殊的隐藏层，即当$l-1=0$时，输入$b_{h'}=x_i$。

#### output layer

输出层则略有不同，与任务目标相关。假设输出单元k的输入为$a_k$，激活值y，则当输出为二分类时，通常输出单元取sigmoid函数：

$$
p(C_1|x)=y=\sigma(a) \\
p(C_2|x)=1-y
$$

其中$C_k$是类别标签。这其实就是一个*logistic regression(logit model)*，若使用z重新编码，z=1代表$C_1$；$z=0$代表$C_2$，则上式可重写为：

$$
p(z|x)=y^z(1-y)^{1-z}
$$

类似的，多分类时使用softmax函数：

$$
p(C_k|x)=y_k=\frac{e^{a_k}}{\sum_{k'=1}^Ke^{a_{k'}}} \\ p(z|x)=\prod_{k=1}^Ky_k^{z_k}
$$

也就是logistic regression在多项上的扩展，即*multinomial logit model*。

#### loss function

策略损失函数这里通常采用最大似然估计(MLE)。

>  关于MLE、MAP、EM我觉得知乎[这里](http://www.zhihu.com/question/19894595)前两个回答特别好，举的例子也很生动。基本上说清楚了它们的区别。当数据量大时，直接从观测样本估计最大可能的分布，即MLE。若估计过程中使用模型过于复杂，难以计算，或者则使用EM优化求解。若样本不足够，一般加入先验以补充信息，即最大后验MAP。

于是，损失函数可以定义为：

$$
\mathcal{L}(x,z)=-\ln{p(z|x)}=\left\{ \begin{array}{rcl}
(z-1)\ln{(1-y)}-z\ln{y} \\
 -\sum_{k=1}^Kz_k\ln{y_k}
\end{array}\right.
$$

注意，实际上的损失函数是x, z, w的共同函数，因为w作为参数，所以在这里忽略掉了。

#### Backword Pass

后向传播实际上是对梯度下降优化算法效率上的一个考虑，因此首先要清楚后向传播最终需要的是损失函数对参数的偏导数，然后就可以按照梯度下降的方法来更新参数w，比如：

$$
\Delta{w_{ij}}=-\alpha \frac{\partial \mathcal{L}(x,z)}{\partial w_{ij}}
$$

直观上来看$\mathcal{L}$是输出值和真实值的误差，因此是y的函数，然后通过y作用到输出单元的输入$a_k$，这样一直从后向前迭代，所以叫后向传播（BP）。BP其实就是**重复的应用chain rule**。我们还是按照输出单元的不同分别来看。首先是二分类：

$$
\frac{\partial \mathcal{L}(x,z)}{\partial y}=\frac{y-z}{y(1-y)} \\
\frac{\partial \mathcal{L}(x,z)}{\partial a}=\frac{\partial \mathcal{L}(x,z)}{\partial y}\frac{\partial y}{\partial a}=y-z
$$

类似的，对于多分类：

$$
\frac{\partial \mathcal{L}}{\partial a_k}
=\sum_{k'=1}^K\frac{\partial \mathcal{L}}{\partial y_{k'}} \frac{\partial y_{k'}}{\partial a_k} \\
=\sum_{k'=1}^K -\frac{z_{k'}}{y_{k'}} (y_k\delta_{kk'}-y_ky_{k'}) \\
=y_k-z_k
$$

其中，$\delta_{kk'}$是一个脉冲函数，只有当$k=k'$时为1，其余为$0$。这样，我们就得到了输出单元的误差。现在我们将误差后向传播至各隐藏层。类似的，我们先定义$\mathcal{L}$对任意单元j输入$a_j$的导数，然后结合chain rule将参数w的梯度改写为：

$$
\Delta{w_{ij}}=-\alpha \frac{\partial \mathcal{L}(x,z)}{\partial w_{ij}}
=-\alpha \frac{\partial \mathcal{L}}{\partial a_j}\frac{\partial a_j}{\partial w_{ij}} \\
=-\alpha \delta_jb_i
$$

其中：

$$
\delta_j \triangleq \frac{\partial \mathcal{L}}{\partial a_j}
$$

这个我们只能从后向前依次求解，对于在最后的隐藏层单元h：

$$
\delta_{h-1} = \frac{\partial \mathcal{L}}{\partial b_{h-1}} \frac{\partial b_{h-1}}{\partial a_{h-1}} \\
=\theta'(a_{h-1}) \sum_{h\in H_{h}} \frac{\partial \mathcal{L}}{\partial a_h} \frac{\partial a_h}{\partial b_{h-1}} \\
=\theta'(a_{h-1}) \sum_{h\in H_{h}} \delta_h w_{(h-1)h}
$$

然后，按照这个公式依次向前求解。

> 特别的，由于这里的求导公式比较复杂，容易出错，我们可以使用symmetrical finite difference技术来验证求解是否正确。其实，这个技术就是高数中导数的定义：
> 
> $$
> \frac{\partial \mathcal{L}}{\partial w{ij}}=\frac{\mathcal{L(w{ij}+\epsilon)-L(w_{ij}-\epsilon)}}{2\epsilon}+\mathcal{O}(\epsilon^2)
> $$
> 
> 其中$\epsilon$不能太小，否则引发浮点计算下溢。

### RNN

与MLP相比，RNN容许了网络中的自连接，这样做的好处是几乎可以以任意精度逼近任何可衡量的序列映射，当然，需要足够的隐藏单元。关键点就是自连接的单元可以一直保持之前的输入，达到“记忆”的目的。

这节以只有一层隐藏单元的RNN为例。为了更直观的理解，下图给出了unfolding的RNN图。

![unfolding_RNN](/public/img/posts/DPLearning/unfolding_RNN.png)

注意，在不同时刻，网络中的权重被重用。



#### forward pass

从上图可以看出，前向传播的时候隐藏单元的输入多了一项：前一时刻隐藏单元的输出。故：

$$
a_h^t=\sum_{i=1}^Iw_{ih}x_i^t+\sum_{h'=1}^H w_{h'h}b_{h'}^{t-1} \\
b_h=\theta(a_h^t)
$$

#### output layer & loss function

输出层则与MLP一样，由当前时刻隐藏层输出经过输出单元产生的激活值：

$$
a_k^t=\sum_{h=1}^Hw_{hk}b_h^t
$$

因此，损失函数与MLP也一样，就不列出来了。

#### backward pass

传统的BP被改良为两种最著名的方法：1）RTRL(Real time recurrent learning)和2）BPTT(back propagation through time)。这里我们只看BPTT，因为一般来说BPTT更为有效。

BPTT与标准的BP一样，也是重复的使用chain rule。只是针对隐藏层输入的不同，添加了下一时刻对当前时刻的影响：

$$
\delta_h^t=\theta'(a_h^t)(\sum_{k=1}^K\delta_kw_{hk}+\sum_{h'=1}^H \delta_{h'}^{t+1}w_{hh'})
$$

权重参数的更新也是一样，回忆不同时刻边的权重始终被重用，所以更新也需要对所有时刻的w更新：

$$
\frac{\partial \mathcal{L}}{\partial w_{ij}}=\sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial a_j^t}\frac{\partial a_j^t}{\partial w_{ij}}
=\sum_{t=1}^T\delta_j^t b_i^t
$$

#### network training

这里有几个trick可以提高训练速度，并且泛化能力更好。

##### Gradient Descent

上面给出的梯度下降公式是最简单最直接的，叫做*steepest descent or just gradient descent*。这个过程一直重复，即沿着梯度的负方向下降一固定的小步，直至满足某个停止准则(stopping criteria)。但是这种方法最大的问题时容易陷入局部最优。其实有更好的做法，这里只介绍一种，通过增加*momentum*增加下降惯性的方法：

$$
\Delta w^n=m\Delta w^{n-1}-\alpha\frac{\partial \mathcal{L}}{\partial w^n}
$$

如果这种梯度更新计算整个数据集，就叫做*batch learning*。这与*online or sequential learning*对应，不是本篇的主题，咱不介绍。注意，在每次迭代训练之前，最好将数据集的顺序打乱，这样也有助于避免局部最优，这样的训练集称为*training epoch*。

##### Generalisation

通常来说，训练集越大，泛化能力越好。但我们的目标是用固定大小的训练集，得到更好的泛化结果。这通常采用下面几种方法：

1. **early stopping**：使用5%～10%的训练集作为验证集，每次训练在验证集外训练，纪录验证集的误差变化，当误差不再下降，则停止训练。这样做是为了防止过拟合；
2. **input noise**：在输入中添加噪声（e.g.高斯分布的白噪声），间接的增大了训练集。问题是噪声在真实中是否存在，以及噪声的方差多大？通常可以使用验证集确定；
3. **weight noise**：在前向和bp的时候为权重加入噪声，注意**更新梯度仍然在原始的梯度**。这样做的好处是可以网络的鲁棒性（原文为简化网络，降低需求的数据量）。

#### Input representation

输入数据最好经过PCA处理，使数据标准化（均值0，标准差1）。这样的输入激活函数更容易接受。其实就是之前提过老师木的解释，sigmoid函数可以使这样的输入在有观测误差时达到最优。

#### Weight initialisation

权值初始化大多数情况并不敏感，可以均匀分布在[-0.1,0.1]，也可以均值为0，标准差1的高斯分布。

### LSTM

有了前面的基础，理解LSTM就简单了很多。LSTM的核心是针对RNN的*vanishing gradient problem*，增加了memory block，用来代替RNN中的每个隐藏单元。下面我们依次介绍。

还是先看个memory block的直观图解以及具有两个memory block的lstm网络。

 ![memory_block](/public/img/posts/DPLearning/memory_block.png)![lstm](/public/img/posts/DPLearning/lstm.png)

#### Vanishing gradient problem

还记得RNN的主要优势是在映射输入序列到输出序列时，可以利用上下文的信息，即可以记忆。然而这种记忆非常有限，因为在网络单元中传递的误差信号将会呈指数级的增长或消失，这种性质就叫做vanishing gradient problem。下面我们从数学的角度看这个问题。

假设误差信号从t-q时刻的节点v传递给t时刻的节点u，利用RNN误差的公式可得：

$$
\frac{\partial \delta_v^{t-q}}{\partial \delta_u^{t}}=\left\{\begin{array}{rcl}
\theta'(a_v^{t-1})w_{uv} & q=1 \\
\theta'(a_v^{t-q})\sum_{l=1}^n \frac{\partial \delta_v^{t-q+1}}{\partial \delta_u^t} w_{lv} & q>1

\end{array}\right.
$$

其中n是一层中神经元的个数。这是一个递归公式，将其展开可得：

$$
\frac{\partial \delta_v^{t-q}}{\partial \delta_u^{t}}=\sum_{l_1=1}^n \dots \sum_{l_{q-1}=1}^n \prod_{m=1}^q \theta'(a_{l_m}^{t-m})w_{l_m l_{m-1}}
$$

定义后面的连乘项为T，则可以看出T有$n^{q-1}$项，因此当$|T|>1$时，误差会指数级增长；而$|T|<1$时，误差会指数级迅速消失。

为了克服这个问题，其实也很简单。先假设仅仅有一个自身相连的神经元，则有：

$$
\delta_j^t=\theta'(a_j^t)\delta_j^{t+1}w_{jj}
$$

若$\theta'w=1$，则误差不发生变化。但是这就表示激活函数$\theta(a)=\frac{a}{w}$是线性函数。因此在memory block内部，令$w=1$。这样就获得了常数误差流，称为CEC(constant error carrrousel)。但这样做的话，该单元的输入既要通过$w_{jj}=1$的权重保持记忆，又要通过$w_{ij}$的权重更新新的输入。为了应对这种权值更新上的冲突，增加了input和output gate。

这样虽然解决了误差呈指数级增长或消失的问题，但是最初的应用中发现，由于新的输入同样带来信息，实际上CEC的状态值是缓慢增加的。于是又引入了forget gate，对CEC状态进行控制，相当于权值$w_{jj}$变成了一个动态的值，在必要时置零，令记忆单元“遗忘”。

> *peephole*是比较受欢迎的一个变化。即将CEC的状态通过图中的虚线输出到input和output gate，影响下一时刻的输入和输出值。

那么现在我们依旧按照前向后向的顺序分析如何利用梯度下降优化求解。

#### Forward pass

LSTM相对于RNN只改动了隐藏单元为memory block，所以这里只给出memory block中涉及的推导，再memory block外的则可以自然的引用RNN对应的部分。为了方便理解，假设仅有一层的H个隐藏单元，下标c标识C个memory cell(个人猜测C＝H，只是为了方便展示peephole)。I个输入单元，K个输出单元，G标识隐藏层所有的输入（包含cell，gate，不区分输入的type，这时为了方面后面使用truncated bptt进行优化，所以没有必要写的很细）。f作为gate的激活函数，g和h作为block输入和输出的激活函数。$\iota,\phi,\omega$分别作为input gate，forget gate和output gate的下标。下面就可以按照传播的顺序进行推导：

首先，input gate和forget gate可以同时确定，它们依赖于输入单元的输入、上一时刻隐藏单元的输出（RNN）、memory cell上一时刻的状态（peephole）：

##### input gate & forget gate (*=[$\iota|\phi$])

$$
a_*^t=\sum_{i=1}^I w_{i*} x_i^t+\sum_{h=1}^H w_{h*} b_h^{t-1}+\sum_{c=1}^C w_{c*} s_c^{t-1} \\
b_*^t=f(a_*^t)
$$

隐藏单元的输入也可以同时确定：

$$
a_c^t=\sum_{i=1}^I w_{ic} x_i^t+\sum_{h=1}^H w_{hc} b_h^{t-1}
$$

这样，就可以计算cell的状态：

$$
s_c^t=b_\phi^t s_c^{t-1}+b_\iota^t g(a_c^t)
$$

于是，计算output gate，它与前两个gate的区别只在于peephole的状态可以使用当前时刻：

$$
a_\omega^t=\sum_{i=1}^I w_{i\omega} x_i^t+\sum_{h=1}^H w_{h\omega} b_h^{t-1}+\sum_{c=1}^C w_{c\omega} s_c^t \\
b_\omega^t=f(a_\omega^t)
$$

最后，计算隐藏单元的输出：

$$
b_c^t=b_\omega^t h(s_c^t)
$$

#### Backward pass

LSTM的后向传播比较复杂，因此一般使用truncated bptt进行优化。我们首先来看一下正常的bptt的结果，然后使用truncated简化计算。

输出单元处和RNN没有什么不同，因此还是直接分析memory block相关的变量：三个gate的输入边的权重参数和memory block输入边的权重参数，但是注意的是可以定义两个辅助变量便于理解。顺序还是采用forward pass相反的方向，目的是求出损失函数对相应单元输入的偏导$\delta$，然后乘以对应边的另一端的输出就得到边权重的更新值。

首先引入第一个辅助变量，隐藏单元输出值的变化：

$$
\epsilon_c^t=\frac{\partial \mathcal{L}}{\partial b_c^t} \\
=\sum_{k=1}^K\frac{\partial \mathcal{L}}{\partial a_k^t} \frac{\partial a_k^t}{\partial b_c^t}+\sum_{g=1}^G\frac{\partial \mathcal{L}}{\partial a_g^t} \frac{\partial a_g^t}{\partial b_c^t} \\
=\sum_{k=1}^K w_{ck}\delta_k^t+\sum_{g=1}^Gw_{cg}\delta_g^{t+1} \\
\approx \sum_{k=1}^K w_{ck}\delta_k^t
$$

> 这里就可以使用truncated技术省略来自下一时刻的第二项。因为所谓truncated是指错误信号到达memory block后就被截断了，并不在沿着时间序列继续传递下去，只停留在memory block内部。实际实验中确实不会对结果有很大的影响。我猜测是由于CEC将误差无损失的截流在单元内部，自然就不该向外传播。

接着来看output gate：

$$
\delta_\omega^t=\frac{\partial \mathcal{L}}{\partial a_\omega^t} \\
=\sum_{c=1}^C \frac{\partial \mathcal{L}}{\partial b_c^t}\frac{\partial b_c^t}{\partial b_\omega^t}\frac{\partial b_\omega^t}{\partial a_\omega^t} \\
=f'(a_\omega^t)\sum_{c=1}^C \epsilon_c^t h(s_c^t)
$$

于是output gate的输入边权重更新梯度为$\delta_\omega^t x_i^t,\delta_\omega^t b_h^{t-1},\delta_\omega^t s_c^t$。下面的更新梯度与此相似，故只求$\delta$。

再来引入第二个辅助变量：

$$
\epsilon_s^t=\frac{\partial \mathcal{L}}{\partial s_c^t} \\
=\frac{\partial \mathcal{L}}{\partial b_c^t} \frac{\partial b_c^t}{\partial h(s_c^t)} \frac{\partial h(s_c^t)}{\partial s_c^t} 
+\frac{\partial \mathcal{L}}{\partial s_c^{t+1}} \frac{\partial s_c^{t+1}}{\partial s_c^t}
+\frac{\partial \mathcal{L}}{\partial a_\omega^t} \frac{\partial a_\omega^t}{\partial s_c^t}
+\frac{\partial \mathcal{L}}{\partial a_\iota^t} \frac{\partial a_\iota^t}{\partial s_c^t}
+\frac{\partial \mathcal{L}}{\partial a_\phi^t} \frac{\partial a_\phi^t}{\partial s_c^t}
\\
=b_w^th'(s_c^t)\epsilon_c^t+b_\phi^{t+1}\epsilon_s^{t+1}+w_{c\omega}\delta_\omega^t+w_{c\iota}\delta_\iota^{t+1}+w_{c\phi}\delta_\phi^{t+1}
$$

这恐怕是bp中最复杂的公式了，我来依次解释下各项。首先，看memory block的图，查看该单元**指向**输出单元的所有路径，没有一条不同的路径就代表一项；然后运用链式法则展开每个路径；就得到后向传播中该单元的梯度$\delta$。这个辅助变量中可以看到后三项来自于cell state对三个gate的监督，即peephole，所以若不采用peephole的方式就可以省略。第二项来自于下一时刻的状态误差，其实是forget gate对当前状态的调节作用；

再继续传播：

##### cell

$$
\delta_c^t =\frac{\partial \mathcal{L}}{\partial a_c^t} \\
=\frac{\partial \mathcal{L}}{\partial s_c^t}\frac{\partial s_c^t}{\partial g(a_c^t)}\frac{\partial g(a_c^t)}{\partial a_c^t} \\
=\epsilon_c^t b_\iota^t g'(a_c^t)
$$

##### forget gate & input gate

$$
\delta_\phi^t =\frac{\partial \mathcal{L}}{\partial a_\phi^t} \\
=\sum_{c=1}^C\frac{\partial \mathcal{L}}{\partial s_c^t}\frac{\partial s_c^t}{\partial b_\phi^t}\frac{\partial b_\phi^t}{\partial a_\phi^t} \\
=f'(a_\phi^t)\sum_{c=1}^C s_c^{t-1}\epsilon_s^t \\

\delta_\iota^t =\frac{\partial \mathcal{L}}{\partial a_\iota^t} \\
=\sum_{c=1}^C\frac{\partial \mathcal{L}}{\partial s_c^t}\frac{\partial s_c^t}{\partial b_\iota^t}\frac{\partial b_\iota^t}{\partial a_\iota^t} \\
=f'(a_\iota^t)\sum_{c=1}^C g(a_c^{t-1})\epsilon_s^t
$$