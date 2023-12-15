# Deep Learning for Computer Vision: Practical Implementations with TensorFlow and Keras

[![Python](https://img.shields.io/badge/Python-3.8.12-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6.0-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-blue.svg)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-blue.svg)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-yellow.svg)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-yellow.svg)](https://scikit-learn.org/)
[![XML](https://img.shields.io/badge/XML-grey.svg)](https://docs.python.org/3/library/xml.etree.elementtree.html)
[![JSON](https://img.shields.io/badge/JSON-grey.svg)](https://www.json.org/)

Welcome to my digital archive documenting an exploration into the topics of deep learning. This repository is a detailed record of my self-study journey, with a focus on hands-on applications using TensorFlow and Keras from scratch. The main objective is to develop a clear understanding of fundamental concepts and techniques in deep learning.

## Overview

In this repository, you'll find a collection of hands-on implementations covering classic Convolutional Neural Networks (CNNs), Transformer models, and object detection. Each model is accompanied by relevant papers, providing a solid foundation for understanding their underlying principles.

## Key Features
- **Implementation Variety:** Explore several models, including VGG, GoogLeNet, ResNet, SqueezeNet, DenseNet, ResNeXt, Res2Net, and more.
- **Extension Modules:** Modules such as Squeeze-and-Excitation Networks (SE block) and Convolutional Block Attention Module (CBAM) to enhance your model architectures.
- **Evaluation Tools:** Evaluate model performance with modules like Gradient-weighted Class Activation Mapping (Gran-CAM) for insightful visualizations.
- **Object Detection:** Study the loss functions and models for You Only Look Once (YOLO), including version 3.
- **Transformer Models:** Explore the realm of vision with models like Vision Transformer (ViT), Compact Convolutional Transformer (CCT), and Compact Vision Transformer (CVT).
- **Utilities and Applications:** Practical tools, data generators, and applications such as image organizers, similarity retrieval, feature vector database generation, and manga dialogue detection with OCR.

## Table of Contents
- [Implementations](#implementations)
  - [Class Diagram](#class-diagram)
  - [CNN Architecture Models](#cnn-architecture-models)
    - [Extension Modules](#extension-modules)
    - [Evaluation Module](#evaluation-module)
  - [Object Detection Models](#object-detection-models)
  - [Transformer Models](#transformer-models)
  - [Other Implementations](#other-implementations)
    - [Data Generators](#data-generators)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Utilities](#utilities)
- [Applications](#applications)

## Implementations

### Class Diagram

<div style="display: flex; justify-content: center;">
  <div style="margin: 10px;">
    <figure style="text-align: center;">
      <img src="https://raw.githubusercontent.com/catptype/DeepLearning-SelfStudy/main/docs/Class%20Diagram.png" style="max-height:auto;width:auto">
    </figure>
  </div>
</div>

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

| Generator Type           | Code Link         |
|--------------------------|-------------------|
| Image Classification     | [Code](https://github.com/catptype/DeepLearning-SelfStudy/tree/main/module/image_classification) |
| Object Detection         | [Code](https://github.com/catptype/DeepLearning-SelfStudy/tree/main/module/object_detection) |

#### Evaluation Metrics

| Metrics                   | Code Link         |
|---------------------------|-------------------|
| Confusion Matrix | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/evaluate/ConfusionMatrix.py) |
| F1 Score | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/metric/F1Score.py) |
| Mean Average Precision (mAP) for object detection | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/metric/MeanAveragePrecision.py) |

#### Utilities

| Utilities                 | Code Link         |
|---------------------------|-------------------|
| Directory Processor | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/utility/DirectoryProcessor.py) |
| Text-based Progress Bar | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/utility/TextProgressBar.py) |

## Applications

This section showcases the practical applications of the deep learning concepts presented in this repository. Explore real-world scenarios and witness how deep learning concepts transform into useful tools and solutions:

- **Image Organizer:** Say goodbye to messy photo folders! This application employs class prediction models to automatically organize your images based on their content labels, making retrieval and browsing easy.

- **Image Feature Vector Database and Similarity Retrieval:** Create a robust database of image features in JSON format. This facilitates efficient similarity searches and image analysis tasks, allowing you to easily find visually similar images from your collection. This application utilizes deep learning models to retrieve images from your collection based on their visual content.

- **Manga Dialogue Detection with OCR:** Extract dialogue from manga images using the deep learning of object detection and Optical Character Recognition (OCR). This application automatically identifies speech bubbles and extracts the text within, enhancing your manga reading experience.

Summary in the table below:

| Applications              | Links         |
|---------------------------|-------------------|
| Image Organizer | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/application/ImageOrganizer.py) [Demo](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/Application%20-%20Image%20Organzier.ipynb)|
| Image Feature Vector Database (JSON format) | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/application/ImageFeatureExtractor.py) [Demo](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/Application%20-%20Image%20Similarity.ipynb) |
| Image Similarity Retrieval | [Code](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/module/application/ImageSimilarity.py) [Demo](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/Application%20-%20Image%20Similarity.ipynb) |
| Manga Dialogue Detection with OCR | [Demo](https://github.com/catptype/DeepLearning-SelfStudy/blob/main/Application%20-%20Manga%20Dialogue%20Detection%20with%20OCR.ipynb) |