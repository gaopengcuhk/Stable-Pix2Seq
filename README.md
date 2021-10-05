# Stable-Pix2Seq
A full-fledged version of Pix2Seq

**What it is**. This is a full-fledged version of Pix2Seq. Compared with [unofficial-pix2seq](https://github.com/gaopengcuhk/Unofficial-Pix2Seq), stable-pix2seq contain most of the tricks mentioned in Pix2Seq like Sequence Augmentation, Batch Repretation, Warmup, Linear decay leanring rate and beam search(to be add later). 

**Difference between Pix2Seq**. In sequence augmentation, we only augment random bounding box while original paper will mix with virual box from ground truth plus noise. Pix2seq also use input sequence dropout to regularize the training process. 

# Usage - Object detection
There are no extra compiled components in Stable-Pix2Seq and package dependencies are minimal,
so the code is very simple to use. We provide instructions how to install dependencies via conda.
First, clone the repository locally:
```
git clone https://github.com/gaopengcuhk/Stable-Pix2Seq.git
```
Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
That's it, should be good to train and evaluate detection models.

(optional) to work with panoptic install panopticapi:
```
pip install git+https://github.com/cocodataset/panopticapi.git
```

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training
To train baseline Stable-Pix2Seq on a single node with 8 gpus for 300 epochs run:
```
python -m torch.distributed.launch --master_port=3141 --nproc_per_node 8 --use_env main.py --coco_path ./coco/ --batch_size 4 --lr 0.0005
```
A single epoch takes 50 minutes on 8 V100, so 300 epoch training
takes around 10 days on a single machine with 8 V100 cards.

**Why slower than DETR and Unofficial-Pix2Seq?**. Stable-Pix2Seq use batch repeat which double the training time. Besides, stable-pix2seq use 1333 image resolution will the time report in unofficial-pix2seq is trained on low resolution 512. 


We train DETR with AdamW setting learning rate using a linear warmup and decay schedule. Due to batch repeat, the real barch size is 64. 
Horizontal flips, scales and crops are used for augmentation.
Images are rescaled to have min size 800 and max size 1333.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.


## Evaluation
To evaluate Stable-Pix2Seq R50 on COCO val5k with multiple GPU run:
```
python -m torch.distributed.launch --master_port=3142 --nproc_per_node 8 --use_env main.py --coco_path ./coco/ --batch_size 4 --eval --resume checkpoint.pth
```

