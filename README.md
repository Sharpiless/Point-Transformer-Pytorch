# Pytorch Implementation of Point Transformer

基于Point Transformers复现点云分割任务

## 论文地址：

[https://arxiv.org/abs/2012.09164](https://arxiv.org/abs/2012.09164)


## 准备数据：
使用连接下载 **ShapeNet** 数据集：[下载地址](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) 

下载完成后解压到 `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

## 训练：

```bash
python train.py
```

## 实验结果：


| Models                   | Accuracy | cat.mIOU | ins.mIOU |
| ------------------------ | -------------- | ------------ | ------------ |
| Point Transformer (paper)      |       None      |     0.837    |    0.866     |
| Point Transformer (our-no quant)  |       0.93456      |     0.79787    |    0.83578     |

# 关注我的公众号：

感兴趣的同学关注我的公众号——可达鸭的深度学习教程：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210127153004430.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)


# 联系作者：

> B站：[https://space.bilibili.com/470550823](https://space.bilibili.com/470550823)

> CSDN：[https://blog.csdn.net/weixin_44936889](https://blog.csdn.net/weixin_44936889)

> AI Studio：[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156)

> Github：[https://github.com/Sharpiless](https://github.com/Sharpiless)

# Reference：

[https://github.com/qq456cvb/Point-Transformers](https://github.com/qq456cvb/Point-Transformers)
