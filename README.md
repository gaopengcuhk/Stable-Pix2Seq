# Stable-Pix2Seq
A full-fledged version of Pix2Seq

**What it is**. This is a full-fledged version of Pix2Seq. Compared with![unofficial-pix2seq](https://github.com/gaopengcuhk/Unofficial-Pix2Seq), stable-pix2seq contain most of the tricks mentioned in Pix2Seq like Sequence Augmentation, Batch Repretation, Warmup and Linear decay leanring rate. 

**Difference between Pix2Seq**. In sequence augmenttaion, we only augment random bounding box while original paper will mix with virual box from ground truth plus noise. Pix2seq also use input sequence dropout to regularize the training. 

