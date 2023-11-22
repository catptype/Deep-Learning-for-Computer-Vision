# Self-Study in Deep Learning with TensorFlow and Keras
A personal repository documenting my deep learning self-study journey with practical implementations using TensorFlow and Keras.

## Table of Contents
- [Introduction](#introduction)
- [Tools, Frameworks, and Libraries](#tools-frameworks-and-libraries)
- [Model Implementation](#model-implementation)
- [Module Implementation](#module-implementation)
- [Other Implementations](#other-implementations)

## Introduction
This repository serves as a digital archive of my journey into deep learning. My primary focus is on practical implementations using TensorFlow and Keras. Through this self-study, I aim to gain a deeper understanding of deep learning concepts and techniques.

## Tools, Frameworks, and Libraries
- **Language:**
  - Python 3.8.12
- **Frameworks:**
  - TensorFlow 2.6.0
  - Keras
- **Libraries:**
  - OpenCV
  - scikit-learn (sklearn)
  - NumPy

## Model Implementation

### CNN Architecture

| Models                    | Paper Link                                      | Code Link         |
|---------------------------|-------------------------------------------------|-------------------|
| VGG                       | [Paper](https://arxiv.org/abs/1409.1556)        | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/VGGModel.py)          |
| GoogLeNet (Inception v1)  | [Paper](https://arxiv.org/abs/1409.4842)        | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/GooLeNetModel.py)          |
| Residual Network (ResNet) | [Paper](https://arxiv.org/abs/1512.03385)       | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/ResnetModel.py)          |
| SqueezeNet                | [Paper](https://arxiv.org/abs/1602.07360)       | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/SqueezeNetModel.py)          |
| DenseNet                  | [Paper](https://arxiv.org/abs/1608.06993)       | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/DenseNetModel.py)          |
| ResNeXt                   | [Paper](https://arxiv.org/abs/1611.05431)       | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/ResNeXtModel.py)          |
| Res2Net                   | [Paper](https://arxiv.org/abs/1904.01169)       | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/Res2NetModel.py)          |

### Object Detection

| Models                   | Paper Link                                      | Code Link         |
|--------------------------|-------------------------------------------------|-------------------|
| You Only Look Once (YOLO) loss function | [Paper](https://arxiv.org/abs/1506.02640)        | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/loss/YOLOLoss.py)          |
| You Only Look Once (YOLO) version 3 | [Paper](https://arxiv.org/abs/1804.02767)        | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/YOLOv3.py)          |

### Transformer

| Models                   | Paper Link                                      | Code Link         |
|--------------------------|-------------------------------------------------|-------------------|
| Vision Transformer (ViT) | [Paper](https://arxiv.org/abs/2010.11929)     | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/VisionTransformer.py) |
| Compact Convolutional Transformer (CCT) | [Paper](https://arxiv.org/abs/2104.05704) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/CompactConvolutionalTransformer.py) |
| Compact Vision Transformer (CVT) | [Paper](https://arxiv.org/abs/2104.05704) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/CompactVisionTransformer.py) |

## Module Implementation

### Extension Modules

| Module                    | Paper Link                                      | Code Link         |
|---------------------------|-------------------------------------------------|-------------------|
| Squeeze-and-Excitation Networks (SE block) | [Paper](https://arxiv.org/abs/1709.01507) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/SE_Module.py) |
| Convolutional Block Attention Module (CBAM) | [Paper](https://arxiv.org/abs/1807.06521) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/CBAM_Module.py) |

### Evaluation module
| Module                    | Paper Link                                      | Code Link         |
|---------------------------|-------------------------------------------------|-------------------|
| Gradient-weighted Class Activation Mapping (Gran-CAM) | [Paper](https://arxiv.org/abs/1610.02391) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/evaluate/GranCAM.py) |

## Other Implementation

### Data Generators
| Data Generators            | Code Link         |
|---------------------------|-------------------|
| Image data generator for image classifcation | [Code](https://github.com/catptype/DeepLearning-SelfStudy/tree/main/module/image_classification) |
| Image data generator for object detection | [Code](https://github.com/catptype/DeepLearning-SelfStudy/tree/main/module/object_detection) |

### Evaluation Metrics
| Metrics            | Code Link         |
|---------------------------|-------------------|
| Confusion Matrix | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/evaluate/ConfusionMatrix.py) |
| F1 Score | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/metric/F1Score.py) |
| Mean Average Precision (mAP) for object detection | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/metric/MeanAveragePrecision.py) |

### Applications and Utilities
| Applications / Utilities  | Code Link         |
|---------------------------|-------------------|
| Image organizer based on class prediction | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/utility/ImageOrganizer.py) |
| Image similarity retrieval | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/utility/ImageSimilarity.py) |
| Image feature vector DB generator (JSON format) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/utility/ImageFeatureExtractor.py) |
| Text-based progress bar | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/utility/TextProgressBar.py) |

