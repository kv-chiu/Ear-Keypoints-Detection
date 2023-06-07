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
CUDA_VISIBLE_DEVICES=1 PORT=8082 nohup python /media/ders/mazhiming/mm/mmhomeworke2/work/mmdetection/tools/train.py \
    /media/ders/mazhiming/mm/mmhomeworke2/data/MMPosehomework/example/rtmdet_tiny_ear.py \
    --work-dir /media/ders/mazhiming/mm/mmhomeworke2/work/result/mmdet > output.log 2>&1 &
```

```python
# 移动到work/mmpose目录
cd /media/ders/mazhiming/mm/mmhomeworke2/work/result/mmpose

# 开始训练
CUDA_VISIBLE_DEVICES=2 PORT=8083 nohup python /media/ders/mazhiming/mm/mmhomeworke2/work/mmpose/tools/train.py \
    /media/ders/mazhiming/mm/mmhomeworke2/data/MMPosehomework/example/rtmpose-s-ear.py \
    --work-dir /media/ders/mazhiming/mm/mmhomeworke2/work/result/mmpose > output.log 2>&1 &
```

```python
# 中途断开，调参数再训
CUDA_VISIBLE_DEVICES=2 PORT=8083 nohup python </Absolute/Path/of/mmpose/tools/train.py> \
    </Absolute/Path/of/data/MMPosehomework/example/rtmpose-s-ear.py> \
    --work-dir </Absolute/Path/of/work/result/mmpose> \
	 > output.log 2>&1 &
	
# 开始训练
CUDA_VISIBLE_DEVICES=2 PORT=8083 nohup python /media/ders/mazhiming/mm/mmhomeworke2/work/mmpose/tools/train.py \
    /media/ders/mazhiming/mm/mmhomeworke2/data/MMPosehomework/example/rtmpose-s-ear.py \
    --work-dir /media/ders/mazhiming/mm/mmhomeworke2/work/result/mmpose \
    --resume /media/ders/mazhiming/mm/mmhomeworke2/work/result/mmpose/epoch_150.pth > output.log 2>&1 &
```

## 推理测试

```python
# mmdet单图推理
CUDA_VISIBLE_DEVICES=1 PORT=8082 python /media/ders/mazhiming/mm/mmhomeworke2/work/mmdetection/demo/image_demo.py \
	/media/ders/mazhiming/mm/mmhomeworke2/data/test_ear.jpg \
    /media/ders/mazhiming/mm/mmhomeworke2/data/MMPosehomework/example/rtmdet_tiny_ear.py \
	--weights /media/ders/mazhiming/mm/mmhomeworke2/work/result/mmdet/20230607_004529/best_coco_bbox_mAP_epoch_200.pth \
	--out-dir /media/ders/mazhiming/mm/mmhomeworke2/work/application/mmdet \
	--device cuda \
	> output.log 2>&1

python /media/ders/mazhiming/mm/mmhomeworke2/work/mmpose/demo/inferencer_demo.py --show-alias

# mmpose单图推理
CUDA_VISIBLE_DEVICES=1 PORT=8082 python /media/ders/mazhiming/mm/mmhomeworke2/work/mmpose/demo/topdown_demo_with_mmdet.py \
	/media/ders/mazhiming/mm/mmhomeworke2/data/MMPosehomework/example/rtmdet_tiny_ear.py \
	/media/ders/mazhiming/mm/mmhomeworke2/work/result/mmdet/20230607_004529/best_coco_bbox_mAP_epoch_200.pth \
    /media/ders/mazhiming/mm/mmhomeworke2/data/MMPosehomework/example/rtmpose-s-ear.py \
	/media/ders/mazhiming/mm/mmhomeworke2/work/result/mmpose/20230607_001345/best_PCK_epoch_140.pth \
	--input /media/ders/mazhiming/mm/mmhomeworke2/data/test_ear.jpg \
	--output-root /media/ders/mazhiming/mm/mmhomeworke2/work/inference/mmpose \
	--save-predictions \
	--device cuda \
	--bbox-thr 0.5 \
    --kpt-thr 0.5 \
    --nms-thr 0.3 \
    --radius 8 \
    --thickness 7 \
    --draw-bbox \
    --draw-heatmap \
    --show-kpt-idx \
	> output.log 2>&1
```
