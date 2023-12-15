# Self-Study in Deep Learning with TensorFlow and Keras
A personal repository documenting my deep learning self-study journey with practical implementations using TensorFlow and Keras.

## Table of Contents
- [Introduction](#introduction)
- [Tools, Frameworks, and Libraries](#tools-frameworks-and-libraries)
- [Class Diagram](#class-diagram)
- [Implementations](#implementations)
  - [CNN Architecture Models](#cnn-architecture-models)
    - [Extension Modules](#extension-modules)
    - [Evaluation Module](#evaluation-module)
  - [Object Dtection Model](#object-detection-models)
  - [Transformer Models](#transformer-models)
  - [Other Implementations](#other-implementations)
    - [Data Generators](#data-generators)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Utilities](#utilities)
- [Applications](#applications)

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
  - JSON

## Class Diagram

## Implementations

### CNN Architecture Models

| Models                    | Paper Link                                      | Code Link         |
|---------------------------|-------------------------------------------------|-------------------|
| VGG                       | [Paper](https://arxiv.org/abs/1409.1556)        | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/VGGModel.py)          |
| GoogLeNet (Inception v1)  | [Paper](https://arxiv.org/abs/1409.4842)        | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/GooLeNetModel.py)          |
| Residual Network (ResNet) | [Paper](https://arxiv.org/abs/1512.03385)       | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/ResnetModel.py)          |
| SqueezeNet                | [Paper](https://arxiv.org/abs/1602.07360)       | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/SqueezeNetModel.py)          |
| DenseNet                  | [Paper](https://arxiv.org/abs/1608.06993)       | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/DenseNetModel.py)          |
| ResNeXt                   | [Paper](https://arxiv.org/abs/1611.05431)       | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/ResNeXtModel.py)          |
| Res2Net                   | [Paper](https://arxiv.org/abs/1904.01169)       | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/Res2NetModel.py)          |

#### Extension Modules

| Module                    | Paper Link                                      | Code Link         |
|---------------------------|-------------------------------------------------|-------------------|
| Squeeze-and-Excitation Networks (SE block) | [Paper](https://arxiv.org/abs/1709.01507) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/extension/SE_Module.py) |
| Convolutional Block Attention Module (CBAM) | [Paper](https://arxiv.org/abs/1807.06521) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/extension/CBAM_Module.py) |

#### Evaluation Module

| Module                    | Paper Link                                      | Code Link         |
|---------------------------|-------------------------------------------------|-------------------|
| Gradient-weighted Class Activation Mapping (Gran-CAM) | [Paper](https://arxiv.org/abs/1610.02391) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/evaluate/GranCAM.py) |

### Object Detection Models

| Models                   | Paper Link                                      | Code Link         |
|--------------------------|-------------------------------------------------|-------------------|
| You Only Look Once (YOLO) loss function | [Paper](https://arxiv.org/abs/1506.02640)        | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/loss/YOLOLoss.py)          |
| You Only Look Once (YOLO) model version 3 | [Paper](https://arxiv.org/abs/1804.02767)        | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/YOLOv3.py)          |

### Transformer Models

| Models                   | Paper Link                                      | Code Link         |
|--------------------------|-------------------------------------------------|-------------------|
| Vision Transformer (ViT) | [Paper](https://arxiv.org/abs/2010.11929)     | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/VisionTransformer.py) |
| Compact Convolutional Transformer (CCT) | [Paper](https://arxiv.org/abs/2104.05704) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/CompactConvolutionalTransformer.py) |
| Compact Vision Transformer (CVT) | [Paper](https://arxiv.org/abs/2104.05704) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/model/CompactVisionTransformer.py) |

### Other Implementations

#### Data Generators
| Data Generators           | Code Link         |
|---------------------------|-------------------|
| Image data generator for image classifcation | [Code](https://github.com/catptype/DeepLearning-SelfStudy/tree/main/module/image_classification) |
| Image data generator for object detection | [Code](https://github.com/catptype/DeepLearning-SelfStudy/tree/main/module/object_detection) |

#### Evaluation Metrics
| Metrics                   | Code Link         |
|---------------------------|-------------------|
| Confusion matrix | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/evaluate/ConfusionMatrix.py) |
| F1 score | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/metric/F1Score.py) |
| Mean Average Precision (mAP) for object detection | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/metric/MeanAveragePrecision.py) |

#### Utilities
| Utilities                 | Code Link         |
|---------------------------|-------------------|
| Directory processor | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/utility/DirectoryProcessor.py) |
| Text-based progress bar | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/utility/TextProgressBar.py) |

## Applications 
| Applications              | Links         |
|---------------------------|-------------------|
| Image organizer based on class prediction | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/application/ImageOrganizer.py) [Demo](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/Application%20-%20Image%20Organzier.ipynb)|
| Image similarity retrieval | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/application/ImageSimilarity.py) [Demo](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/Application%20-%20Image%20Similarity.ipynb) |
| Image feature vector DB generator (JSON format) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/application/ImageFeatureExtractor.py) [Demo](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/Application%20-%20Image%20Similarity.ipynb) |
| Manga Dialogue Detection with OCR | [Demo](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/Application%20-%20Manga%20Dialogue%20Detection%20with%20OCR.ipynb) |