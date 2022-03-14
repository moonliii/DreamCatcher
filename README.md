##  1. Similarity

> 因为过于复杂的算法涉及到专业的数字图像处理的知识，学习难度较大，而我们的功能对精确度要求不是很大，故考虑尽量简单但符合一定准确度需求的算法

在小程序最初的试宣传阶段，开放主要功能：即画图功能。为了达到一定的趣味性和宣传效果，我们附加了好友之间图画相似度的功能。

#### 概述

关于图片相似度的实现，研究者们提出了很多算法：

- 基于颜色分布的算法：生成图片的颜色分布的直方图，但直方图过于简单，只能捕捉颜色信息的相似性，捕捉不到更多的信息。只要颜色分布相似，就会判定二者相似度较高，准确度不高。

- 哈希算法类，这类算法包含了均值哈希算法、插值哈希算法、感知哈希算法。该算法的基本原理是：对每张图片生成一个“指纹”字符串，然后比较不同图片的指纹。结果越接近，就说明图片越相似。哈希算法侧重于图片整体的相似度，但我们需要对图画的主体内容敏感，因此不适于我们的应用场景。

- sift算法（尺度不变特征转换），在空间尺度中寻找极值点，并提取出其位置、尺度、旋转不变量。sift在图像的不变特征提取方面有很大的优势，但我们的图像像素不高且内容较简单，相比之下它的计算过程过于繁琐。

- 最终选择同样是基于图像局部性特征的方法：Hu不变矩。图像几何距是图像的几何特征，高阶几何距中心化之后具有特征不变性。考虑利用Hu不变矩这个简单而且有效的特征描述子，参考openCV中关于轮廓匹配`matchShapes`的实现，来得到基于Hu矩计算两张图片的相似度。

  > 由Hu矩组成的特征量对图片进行识别，优点就是速度很快，缺点是识别率比较低。Hu不变矩一般用来识别**图像中大的物体**，对于物体的形状描述得比较好，图像的纹理特征不能太复杂
  >
  > [使用Hu矩进行匹配](https://blog.csdn.net/xizero00/article/details/7448070)

  > 参考：[Moment矩,轮廓特征,轮廓匹配,形状匹配](https://blog.csdn.net/KYJL888/article/details/85060883)

#### 具体实现

> OepnCV

- base64流转opencv图片

  ```python
  def base64_to_image(base64_code):
      # base64解码
      img_data = base64.b64decode(base64_code)
      # 转换为np数组
      img_array = np.fromstring(img_data, np.uint8)
      # 转换成opencv可用格式
      img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
      return img
  ```

- 图片预处理

  - 高斯模糊 - [平滑图像](http://codec.wang/#/opencv/basic/10-smoothing-images)
  - 转灰度图
  - 图像二值化

- 得到Hu矩

  - `cv2.moments(binary_image)`
  - `cv2.HuMoments(moments)`

- 相似度

  - 参考`matchShapes`[源码](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/matchcontours.cpp)对Hu矩进行处理，得到余弦相似度
  - 最后对得到的结果有一个“经验处理”：
    - 对异常高的相似度减去一个距离因子（`CV_CONTOURS_MATCH_I2`）
    - 负值（->正值）意味着相似度很低



## 2.图像分类

要做通过特定主题的涂鸦来判断心理状况，然后再进一步推荐歌曲。

这其中很核心的一步就是做图像分类，虽然网上已经有很多能达到高准确度的模型，但要想适用于我们的情形，还是要做部分更改的。

首先我们的目标是对**特定主题**的涂画来进行分类，特定主题指的是房树人（来源于**心理学测试**），我们花了很大的精力去找数据集，但大多数都是那种线下画出来然后拍的照片，我们需要的是在手机屏幕上直接涂鸦的效果，所以想干脆我们自己收集想要的数据。然后我们单独开放小程序画画的功能，让组内的同学帮忙宣传，一方面用于我们应用的前期宣传，另一方面是为了收集数据，然后让心理学专业的同学手动对这些原生图片分类。

但是也许我们的宣传不到位，收集到的图片：一是数据比较少；二是各类的比例不平衡

> “面对真实的、未加工过的数据时，你会马上注意到，这些数据要嘈杂且不平衡得多”



#### Data Augmentation

- [处理不平衡的数据](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650718717&idx=1&sn=85038d7c906c135120a8e1a2f7e565ad&scene=0#wechat_redirect)

  - [处理不平衡数据](https://zhuanlan.zhihu.com/p/441429637)

- [数据增强](https://www.zhihu.com/question/319291048)

  - 有监督的数据增强

    有监督数据增强，即**采用预设的数据变换规则，在已有数据的基础上**进行数据的扩增

    - 单样本数据增强
      - 几何变换类：翻转、旋转、裁剪、变形、缩放等
      - 颜色变换类：噪声、模糊、颜色变换、擦除、填充等
    - 多样本数据增强
      - **SMOTE**人工合成新样本来处理样本不平衡问题

  - 无监督数据增强

    无监督的数据增强方法包括两类：

    1.  通过模型学习数据的分布，随机生成与训练数据集分布一致的图片，代表方法GAN
    2.  通过模型，学习出适合当前任务的数据增强方法，代表方法AutoAugment



- 使用

  1. [BorderlineSMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html)（Synthetic Minority Oversampling TEchnique：合成少数类过采样技术） 解决数据集imbalance问题

     > 通过在已有的样本间插值来创造新的少数类样本

  2. [imgaug](https://imgaug.readthedocs.io/en/latest/source/examples_basics.html)

     > 使用了翻转、高斯模糊、强弱对比度、仿射变换等，避免颜色变换（心理学上对画图的情感分类与颜色相关）

- 尝试使用GAN数据增强

  - 结果：生成的都很模糊
  - [用GAN来做数据增强](https://zhuanlan.zhihu.com/p/353430409)



#### 步骤

- 数据增强 & 预处理

  - imgaug扩大数据集（20倍数据量）
  - to_white,  resize，normalize
  - **使用convolutional autoencoder降维** 
  - BorderlineSMOTE处理数据不平衡问题

- classification

  > 尝试resnet-like model & vgg-like model（自己搭建）

  - **注意:** 激活函数和损失函数一定要匹配！
    > 问题：训练时loss不发生变化，accuracy一直为0.5

    二分类：激活函数sigmoid，损失函数binary_crossentropy

    多分类：激活函数softmax，损失函数categorical_crossentropy

    [分析](https://blog.csdn.net/qq_35599937/article/details/105608354)



#### 更新⭐

- 收集一定周期的数据，进行预处理（to_white&resize&normalize），更新convolutional autoencoder；将数据降维，再更新resnet-like model



#### 简化：不经过降维...

