# botify-engine

根据自定义规则生成数据集，作为NER和classifier的训练集，效果如下：

[![Pasted Graphic 1.jpg](https://i.loli.net/2018/08/04/5b65bff9f27e8.jpg)](https://i.loli.net/2018/08/04/5b65bff9f27e8.jpg)

[![Pasted Graphic.jpg](https://i.loli.net/2018/08/04/5b65bffa00e13.jpg)](https://i.loli.net/2018/08/04/5b65bffa00e13.jpg)

## Usage

```
python scholar_simulator.py
```

## File orgnization

```
|- train.py 
|- debug.py 
|- [dir] data (word library)
|- [dir] eval (help tools when creating)
|- [dir] features (word feature)
```