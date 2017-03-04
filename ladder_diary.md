# 关于ladder的实验 （ubuntu16.04）

## i、关于注意事项

### 1、在用fuel-download的时候它说 import libgfortran 出错
>在安装ladder的时候不要用 conda env create -f environment.yml ，也不要按照作者的教程来安装theano,fuel,blocks,而要按以下的步骤走
<pre><code>
conda install theano ###关于theano更详细的配置在第二点
pip install git+git://github.com/mila-udem/blocks.git@release-0.2
</code></pre>

### 2、ubuntu16.04的theano安装
>安装theano http://www.linuxidc.com/Linux/2016-09/135528.htm  
>安装cnmem http://www.cnblogs.com/ZJUT-jiangnan/p/5532724.html

### 3、选择数据库的运行方式(--dataset参数)(此处选了mnist)
<pre><code>
python run.py train --dataset mnist --encoder-layers 1000-500-250-250-250-10 --decoder-spec gauss --denoising-cost-x 1000,10,0.1,0.1,0.1,0.1,0.1 --labeled-samples 100 --unlabeled-samples 60000 --seed 1 -- mnist_100_full
</code></pre>

### 4、遇到AttributeError: 'numpy.float32' object has no attribute 'owner'
>按照以下教程设一下anaconda的environment就可以了，然后 activate一下
>详见 https://github.com/CuriousAI/ladder/issues/17

### 5、fuel-convert报错ImportError: libgfortran.so.1: cannot open shared object file: No such file or directory
>不要在activate ladder的情况下使用fuel-convert

### 6、遇到float32出错之类的问题
>在~/.theano中令floatx=float32
>把pip的blocks和fuel模块都删了，运行
```
pip install git+git://github.com/mila-udem/blocks.git@release-0.2
```

## ii、关于实验的速度
### 1、700组数据用于train，100组用于test，耗时11分钟左右(titain x),错误率27%。
>具体代码如下
```
python run.py train --dataset mnist --encoder-layers 1000-500-250-250-250-10 --decoder-spec gauss --denoising-cost-x 1000,10,0.1,0.1,0.1,0.1,0.1 --labeled-samples 100 --unlabeled-samples 600 --batch-size 100 --valid-batch-size 100 --valid-set-size 1000 --seed 1 -- mnist_100_full
```
>结果如下
```
valid_final_error_rate_clean 27
Took 11.8 minutes
```
>log文件为ladder/result/mnist_100_full21/log.txt
