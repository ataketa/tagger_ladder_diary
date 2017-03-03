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

> *  2.规定系数更新的算法（比如说随机梯度下降）并编译。在最后的主程序中会构造一个叫MainLoop类的对象，这个MainLoop构造的过程中，会调用theano的function()函数，这个函数会把“网络结构+系数更新算法”这整个东西进行编译（因为python原本是跑一句编译一句的，所以会比较慢，function()会把要跑的东西都编译完之后，一起跑，所以会快一点。

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
>to be continue…...
