# caffe2-mobilenet
## Intro
This is a [Caffe2](https://github.com/caffe2/caffe2) implementation of Google's MobileNets.For details, please read the original paper:  
["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" by Andrew G. Howard et. al. 2017](https://arxiv.org/pdf/1704.04861.pdf)  
This is refer to [shicai/MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe). My code is not elegant.

## Loss and accuracy
I provide a pretrained MobileNet model on cifar10, because of smaller image size than imagenet, I reduce some downsampling operation, and got 89+% accuracy.

## How to Use

## Citation
```bash
@article{Howard2017mobilenet,
  Author = {Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam},
  Journal = {arXiv preprint arXiv:1704.04861},
  Title = {MobileNets: Efficient Convolutional Neural Networks for Mobile Vision},
  Year = {2017}
}
```
