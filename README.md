## 创建环境

```shell
# 创建conda环境
conda create -n mmwork python==3.8.16
conda activate mmwork
conda install mamba

# 安装pytorch（根据个人硬件配置，从pytorch选择下载指令）
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 通过openmim安装mmcv和mmdet
mamba install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

# 解压数据集
cd mm/mmhomeworke2
mkdir data
mkdir -p work/result/mmdet
mkdir -p work/result/mmpose
unzip 'data/MMPosehomework.zip' -d 'data/MMPosehomework'
mv 'data/MMPosehomework/样例config配置文件' 'data/MMPosehomework/example'
unzip 'data/MMPosehomework/Ear210_Dataset_coco.zip' -d 'data/MMPosehomework/Ear210_Dataset_coco'

# 准备mmpose和mmdet
cd work
git clone git@github.com:open-mmlab/mmpose.git
git clone git@github.com:open-mmlab/mmdetection.git
cd mmpose
pip install -v -e .
# 以下两行为了避免numpy报错
pip uninstall xtcocotools -y
pip install git+https://github.com/jin-s13/xtcocoapi
# 也是为了解决报错
mamba install scipy
# 同样为了解决报错
pip install -U albumentations --no-binary qudida,albumentations
```

## 修改config文件

打开`rtmdet_tiny_ear.py`和`rtmpose-s-ear.py`

修改以下内容：

```python
data_root = </Absolute/Path/of/Ear210_Keypoint_Dataset_coco/>
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 8
```

## 训练模型

```python
# 移动到work/mmdetection目录
cd /media/ders/mazhiming/mm/mmhomeworke2/work/result/mmdet

# 开始训练
CUDA_VISIBLE_DEVICES=1 PORT=8082 nohup python </Absolute/Path/of/mmdetection/tools/train.py> \
    </Absolute/Path/of/data/MMPosehomework/example/rtmdet_tiny_ear.py> \
    --work-dir </Absolute/Path/of/work/result/mmdet> > output.log 2>&1 &
```

```python
# 移动到work/mmpose目录
cd /media/ders/mazhiming/mm/mmhomeworke2/work/result/mmpose

# 开始训练
CUDA_VISIBLE_DEVICES=2 PORT=8083 nohup python </Absolute/Path/of/mmpose/tools/train.py> \
    </Absolute/Path/of/data/MMPosehomework/example/rtmpose-s-ear.py> \
    --work-dir </Absolute/Path/of/work/result/mmpose> > output.log 2>&1 &
```

```python
# 中途断开，调参数再训
CUDA_VISIBLE_DEVICES=2 PORT=8083 nohup python </Absolute/Path/of/mmpose/tools/train.py> \
    </Absolute/Path/of/data/MMPosehomework/example/rtmpose-s-ear.py> \
    --work-dir </Absolute/Path/of/work/result/mmpose> \
	--resume </Absolute/Path/of/pth> > output.log 2>&1 &
```

## 评估结果

### MMdet (epoch200)

>  Average Precision  (AP) @[ IoU=0.50:0.95 | area=  all | maxDets=100 ] = 0.730
>
>  Average Precision  (AP) @[ IoU=0.50    | area=  all | maxDets=100 ] = 0.962
>
>  Average Precision  (AP) @[ IoU=0.75    | area=  all | maxDets=100 ] = 0.936
>
>  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
>
>  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
>
>  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.730
>
>  Average Recall   (AR) @[ IoU=0.50:0.95 | area=  all | maxDets=  1 ] = 0.764
>
>  Average Recall   (AR) @[ IoU=0.50:0.95 | area=  all | maxDets= 10 ] = 0.769
>
>  Average Recall   (AR) @[ IoU=0.50:0.95 | area=  all | maxDets=100 ] = 0.769
>
>  Average Recall   (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
>
>  Average Recall   (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
>
>  Average Recall   (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.769
>
> 06/04 21:48:25 - mmengine - INFO - bbox_mAP_copypaste: 0.730 0.962 0.936 -1.000 -1.000 0.730
>
> 06/04 21:48:25 - mmengine - INFO - Epoch(val) [200][6/6]   coco/bbox_mAP: 0.7300  coco/bbox_mAP_50: 0.9620  coco/bbox_mAP_75: 0.9360  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.7300  data_time: 1.0237  time: 1.0675

### MMpose (epoch150)

>  Average Precision  (AP) @[ IoU=0.50:0.95 | area=  all | maxDets= 20 ] =  0.575
>
>  Average Precision  (AP) @[ IoU=0.50    | area=  all | maxDets= 20 ] =  1.000
>
>  Average Precision  (AP) @[ IoU=0.75    | area=  all | maxDets= 20 ] =  0.692
>
>  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
>
>  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.575
>
>  Average Recall   (AR) @[ IoU=0.50:0.95 | area=  all | maxDets= 20 ] =  0.610
>
>  Average Recall   (AR) @[ IoU=0.50    | area=  all | maxDets= 20 ] =  1.000
>
>  Average Recall   (AR) @[ IoU=0.75    | area=  all | maxDets= 20 ] =  0.762
>
>  Average Recall   (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
>
>  Average Recall   (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.610
>
> 06/04 22:51:42 - mmengine - INFO - Evaluating PCKAccuracy (normalized by ``"bbox_size"``)...
>
> 06/04 22:51:42 - mmengine - INFO - Evaluating AUC...
>
> 06/04 22:51:42 - mmengine - INFO - Evaluating NME...
>
> 06/04 22:51:42 - mmengine - INFO - Epoch(val) [160][6/6]   coco/AP: 0.574637  coco/AP .5: 1.000000  coco/AP .75: 0.691982  coco/AP (M): -1.000000  coco/AP (L): 0.574637  coco/AR: 0.609524  coco/AR .5: 1.000000  coco/AR .75: 0.761905  coco/AR (M): -1.000000  coco/AR (L): 0.609524  PCK: 0.917234  AUC: 0.067800  NME: 0.056621  data_time: 4.895422  time: 4.944468



**MMpose效果不理想，待炼丹完成……**



## 遇到的坑

### 环境配置

> ```python
> # 以下两行为了避免numpy报错
> pip uninstall xtcocotools -y
> pip install git+https://github.com/jin-s13/xtcocoapi
> # 也是为了解决报错
> mamba install scipy
> # 同样为了解决报错
> pip install -U albumentations --no-binary qudida,albumentations
> ```

从上往下依次来自

1. [Numpy error · Issue #2195 · open-mmlab/mmpose · GitHub](https://github.com/open-mmlab/mmpose/issues/2195)
2. 自行解决
3. https://github.com/open-mmlab/mmpose/pull/1184

### config

MMpose示例config的默认参数中，CosineAnnealingLR开始的epoch偏晚（150epoch），调早一些效果可能更好
