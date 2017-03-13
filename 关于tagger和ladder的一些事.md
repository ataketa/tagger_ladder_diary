# 关于tagger和ladder的代码一些事

### 一、总述

>tagger和ladder主要基于以下体系实现：
* theano：负责主要的常用计算。(http://www.deeplearning.net/software/theano/tutorial/index.html)
* blocks：用于管理网络的运行。(http://blocks.readthedocs.io/en/latest/)
* fuel：用于数据的IO。 (http://fuel.readthedocs.io/en/latest/)

### 二、算法实现部分

>	theano和blocks（其实blocks相当于是theano的一种封装，所以tagger和ladder主要调用blocks）的使用思路大概可以分成3步:
> *  1.写好总的计算表达式（也就是网络结构和cost），特别是cost。比如说tagger.py和ladder.py的apply()函数，就是把各种各样的变量用符号关联起来。在最后的主程序tagger_exp.py或者run.py中可以见到它首先会调用tagger和ladder的构造函数（初始化函数），而这个函数里面，就会调用apply()。

>>```python
def apply(self):
    """ Build the whole Tagger computation graph """
    x = prepend_empty_dim(self.x)
    x_only = prepend_empty_dim(self.x_only)
    y = self.y
    ....
```

> *  2.规定系数更新的算法（比如说随机梯度下降）并编译(可以看到，前面定义好的cost，这个时候作为参数填入了生成GradientDescent的类中)。在最后的主程序中会构造一个叫MainLoop类的对象，这个MainLoop构造的过程中，会调用theano的function()函数，这个函数会把“网络结构+系数更新算法”这整个东西进行编译（因为python原本是跑一句编译一句的，所以会比较慢，function()会把要跑的东西都编译完之后，一起跑，所以会快一点。

>>```python
step_rule = Adam(learning_rate=self.p.lr)
training_algorithm = GradientDescent(
    cost=self.tagger.total_cost, parameters=to_train, step_rule=step_rule,
    on_unused_sources='warn',
    theano_func_kwargs={'on_unused_input': 'warn'}
)
...
```

>>```python
main_loop = MainLoop(
    training_algorithm,
    # Datastream used for training
    self.streams['train'],
    model=Model(self.tagger.total_cost),
    extensions=[
        FinishAfter(after_n_epochs=self.p.num_epochs),
        SaveParams(self.p.get('save_freq', 0), self.tagger, self.save_dir, before_epoch=True),
        DataStreamMonitoring(
            valid_params.values(),
            self.streams['valid'],
            prefix="valid"
        ),
        FinalTestMonitoring(
            test_params.values(),
            self.streams['train'],
            {'valid': self.streams['valid'], 'test': self.streams['test']},
            after_training=True
        ),
        TrainingDataMonitoring(
            train_params.values(),
            prefix="train", after_epoch=True
        ),
        SaveExpParams(self.p, self.save_dir, before_training=True),
        Timing(after_epoch=True),
        ShortPrinting(short_prints, after_epoch=True),
    ])
    ....
```

> *  3.走你。

>>```python
     main_loop.run()
```

### 三、数据接口
>ladder实验中用到了两个数据库MNIST，CIFAR10,都是通过fuel的接口（或者说类）来调用的。  
>首先在最开头import了这两个数据库的类

>```python
from fuel.datasets import MNIST, CIFAR10
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer
...
```

>之后又在setup_data()中调用。先选择用哪个类。

>```python
def setup_data(p, test_set=False):
    dataset_class, training_set_size = {
        'cifar10': (CIFAR10, 40000),
        'mnist': (MNIST, 50000),
    }[p.dataset]
    ...
```

>之后填入参数["train"]，生成一个实例train_set

>```python
    train_set = dataset_class(["train"])
    ...
    d = AttributeDict()
    # Choose the training set
    d.train = train_set
    d.train_ind = all_ind[:training_set_size]
    ...
```

>而MNIST和CIFAR10类，在fuel的文档中描述如下  

>> <h2>CIFAR10<a class="headerlink" href="#module-fuel.datasets.cifar10" title="Permalink to this headline">¶</a></h2>
<dl class="class">
<dt id="fuel.datasets.cifar10.CIFAR10">
<em class="property">class </em><code class="descclassname">fuel.datasets.cifar10.</code><code class="descname">CIFAR10</code><span class="sig-paren">(</span><em>which_sets</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference external" href="https://github.com/mila-udem/fuel/blob/master/fuel/datasets/cifar10.py#L6"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#fuel.datasets.cifar10.CIFAR10" title="Permalink to this definition">¶</a></dt>
<dd><p>Bases: <code class="xref py py-class docutils literal"><span class="pre">fuel.datasets.hdf5.H5PYDataset</span></code></p>
<p>The CIFAR10 dataset of natural images.</p>
<p>This dataset is a labeled subset of the <code class="docutils literal"><span class="pre">80</span> <span class="pre">million</span> <span class="pre">tiny</span> <span class="pre">images</span></code>
dataset [TINY]. It consists of 60,000 32 x 32 colour images in 10
classes, with 6,000 images per class. There are 50,000 training
images and 10,000 test images [CIFAR10].</p>
...
>
>><h2>MNIST<a class="headerlink" href="#module-fuel.datasets.mnist" title="Permalink to this headline">¶</a></h2>
<dl class="class">
<dt id="fuel.datasets.mnist.MNIST">
<em class="property">class </em><code class="descclassname">fuel.datasets.mnist.</code><code class="descname">MNIST</code><span class="sig-paren">(</span><em>which_sets</em>, <em>**kwargs</em><span class="sig-paren">)</span><a class="reference external" href="https://github.com/mila-udem/fuel/blob/master/fuel/datasets/mnist.py#L7"><span class="viewcode-link">[source]</span></a><a class="headerlink" href="#fuel.datasets.mnist.MNIST" title="Permalink to this definition">¶</a></dt>
<dd><p>Bases: <code class="xref py py-class docutils literal"><span class="pre">fuel.datasets.hdf5.H5PYDataset</span></code></p>
<p>MNIST dataset.</p>
<p>MNIST (Mixed National Institute of Standards and Technology) [LBBH] is
a database of handwritten digits. It is one of the most famous
datasets in machine learning and consists of 60,000 training images
and 10,000 testing images. The images are grayscale and 28 x 28 pixels
large. It is accessible through Yann LeCun’s website [LECUN].</p>
...
>
>可见他们的基类都是H5PYDataset。也就是说，我们只要把关于如何把自己的数据，也包装成H5PYDataset的实例（这个在fuel的教程上有说），就几乎可以直接使用ladder中各种关于数据输入以及预处理的代码了（还有一点点会根据不同数据库而定制的代码，还是要自己写的）
