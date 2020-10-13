# 一、图形迭代：
（一）环境配置：  
（1）在pycharm上：python 3.7.x,tensorflow 2.x,numpy等必要的库  
（2）colab：必要的库都具备了  
（二）实践过程：  
（1）在pycharm上：直接运行train.py文件  
# 可以自己修改setting上的文件，修改自己的内容图片和风格图片，内容损失和风格损失，查看不同权重下的迭代效果
（2）Colab上：  
1.加载文件夹到硬盘中：  
from google.colab import drive  
drive.mount('/content/drive')  

2.更改运行目录：  
import os  
os.chdir("/content/drive/My Drive/图形迭代") # 更改成你自己的文件夹路径  

3.运行train.py文件进行图形迭代：  
! python train.py  
# 可以查看output文件夹下查看图形迭代的效果图

# 二、快速风格迁移：
（一）环境配置：  
（1）在pycharm上：python 2.7.x,tensorflow >=1.0,pyyaml,numpy等必要的库  
（2）Colab上：需要用一个命令将代码运行环境调到tensorflow 1.x版本,你可能会用到的有：  
1. %tensorflow_version 1.x  
2. import tensorflow as tf  
3. tf.__version__  
（二）数据集：COCO2014  
（三）实践过程：  
（1）在pycharm上：  
1.训练模型：直接运行train.py  
2.模型应用：直接运行eval.py文件  
（2）在colab上：  
1.将自己的风格图片放在img文件夹中，例如nahan.jpg  

2.配置yml文件：复制conf文件夹中的任一yml文件并重命名为风格图片名.yml（例如我重命名为nahan.yml）  

3.将第2行、第3行的绿框处改为你的风格图片名。将第8行的红框处改为你的风格权重。（风格权重的数值需要自行测试得出最佳，  
可以借助图形迭代的代码去获取最佳，因为图形迭代的时间比较短，能够较快测出最佳的风格权重的数值，  
我在这里测试的数据是220，在下面的内容会有两个一样的图片，设置不同的参数，得出不同的模型的效果图片的对比）。代码如下：  

# Basic configuration  
style_image:img/nahan.jpg  
naming:"nahan"  
model_path:models  

# weight of the loss  
content_weight:1.0  
style_loss:220.0  
tv_weight:0.0  

4.加载文件夹到硬盘中：  
from google.colab import drive  
drive.mount('/content/drive')  

5.更改运行目录：  
import os  
os.chdir("/content/drive/My Drive/Fast-style") # 更改成你自己的文件夹路径  

6.训练模型：  
! python train.py -c conf/nahan.yml # 替换为你配置的yml文件，该过程大概需要6个小时左右  

7.应用模型：  
! python eval.py --model_file ./models/nahan/fast-style-model.ckpt-done --image_file img/wave.jpg   
# 可以将模型替换为任意你训练的模型，以及应用的图片.jpg可以更改，运行时间大概为2-3秒钟时间  
# 图片生成可以在generated文件夹下查看  

# 可以通过更改以下代码更改图片保存名称及路径：  
generated_file = 'generated/res.jpg'  # 可以更改为generated/huashi.jpg  

# 三、CycleGAN  
(一)CycleGAN-train.py  
1.配置环境：(1)python3.7,tensorflow2.1  
           (2)colab
2.数据集：采用网上的monet2photo和vangogh2photo（下载后需要置换A，B才能实现图片转风格）或者coco2014  
3.设置数据集路径：path变量名分别为：train_photo,train_monet,test_photo,test_monet  
4.迭代次数：epoch=40，batch_size=1  
5.保存训练模型checkpoint：设置路径ckpt= "./checkpoints/train",最多只保留最新训练的两个模型max_to_keep=2  

(二)训练完成后重新加载模型CycleGAN-test.py  
6.加载模型，设置路径为第五点的ckpt路径，使用testC进行测试。  


