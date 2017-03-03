# ubuntu16.04+Anaconda2+theano运行tagger实验

## i、实验注意事项
### 1.关于安装
>1. 首先安装好ladder，并用一组比较小的数据测试一下，确认ladder可以正常使用，详见ladder_diary.md。
>2. 然后下载tagger的代码，注意，不要使用--recursive选项
```
git clone https://github.com/CuriousAI/tagger.git
```
>3. 把刚刚调试成功的ladder复制到tagger/ladder中
>4. 安装tagger（详见https://github.com/CuriousAI/tagger）
```
bash ./install.sh
```

### 2.在publication_visualization.ipynb中改变分组（n_groups）之后出错
>原因是在画图的辅助脚本analyze.py中只定义了4种颜色。可以更改analyze.py中的第8行，自己增加或者减少颜色，使得颜色的个数和n_groups一致，因为后面有个点乘，维数要对上才行。
```
GROUP_COLORS = np.array([[0.122, 0.031, 0.408], [0.592, 0.016, 0], [0, 0.467, 0.031], [0.592, 0.49, 0]])
```


### 3.关于使用theano的pydotprint功能,来画网络结构图
>安装graphviz
```
sudo apt-get upgrade -f graphviz
```
>安装graphviz封装
```
conda install graphviz
```
>安装pydot
```
conda install -c anaconda pydot=1.0.28
```
>参考网址
https://github.com/rainyear/lolita/issues/30
http://graphviz.org/content/installation-ubuntu
https://anaconda.org/anaconda/pydot

>但是其实后来失败了，以下两个函数都报错了，搞不清楚原因
```
theano.d3viz.d3viz
theano.printing.pydotprin
```
报错的内容是
```
*** InvocationException: Program terminated with status: -11. stderr follows: []
```

