# Self-Study in Deep Learning with TensorFlow and Keras
A personal repository documenting my deep learning self-study journey with practical implementations using TensorFlow and Keras.

## Table of Contents
- [Introduction](#introduction)
- [Model implementation](#model-implementation)
- [Module implementation](#module-implementation)
- [Other Implementation](#other-Implementation)

## Introduction
This repository serves as a digital archive of my journey into deep learning. My primary is focusing on practical implementations using TensorFlow and Keras. Through this self-study, I aim to gain a deeper understanding of deep learning concepts and techniques.

## Model implementation
In this section, I present an overview of the deep learning models that I have explored and personally implemented. Each model is accompanied by an URL link to its respective paper for reference.

### CNN Architecture
- [VGG](https://arxiv.org/abs/1409.1556)
- [GoogLeNet (Inception v1)](https://arxiv.org/abs/1409.4842)
- [Residual Network (ResNet)](https://arxiv.org/abs/1512.03385)
- [SqueezeNet](https://arxiv.org/abs/1602.07360)
- [DenseNet](https://arxiv.org/abs/1608.06993)
- [ResNeXt](https://arxiv.org/abs/1611.05431)
- [Res2Net](https://arxiv.org/abs/1904.01169)

### Object Detection
- [YOLOv3](https://arxiv.org/abs/1804.02767) and its loss function idea from [YOLOv1](https://arxiv.org/abs/1506.02640)

### Transformer
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [Compact Transformers: CVT & CCT](https://arxiv.org/abs/2104.05704)

## Module implementation
In addition to the core models, I have explored extension modules that enhance model performance and capabilities. These modules are essential for creating state-of-the-art deep learning architectures.

### Extension modules
- [Squeeze-and-Excitation Networks (SE block)](https://arxiv.org/abs/1709.01507)
- [Convolutional Block Attention Module (CBAM)](https://arxiv.org/abs/1807.06521) 

### Evaluation module
- [Gradient-weighted Class Activation Mapping (Gran-CAM)](https://arxiv.org/abs/1610.02391)

## Other Implementation

### Data Generator
- Image data generator for image classifcation: [[Code](https://github.com/catptype/DeepLearning-SelfStudy/tree/main/module/image_classification)]
- Image data generator for object detection: [[Code](https://github.com/catptype/DeepLearning-SelfStudy/tree/main/module/object_detection)]

### Evaluation Metrics
- F1 score: [[Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/metric/F1Score.py)]
- Mean Average Precision (mAP) for object detection: [[Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/metric/MeanAveragePrecision.py)]