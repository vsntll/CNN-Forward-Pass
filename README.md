# Convolutional Neural Network Forward Pass Project

## Overview

This project involves implementing the forward pass of a Convolutional Neural Network (CNN) for object recognition using Matlab. The goal is to gain practical experience with Matlab programming, multi-dimensional arrays, and essential CNN operations such as convolution, normalization, maxpooling, ReLU activation, fully connected layers, and softmax.

The CNN is designed to classify 32x32 color images from the CIFAR-10 dataset into 10 object classes by applying a chain of 18 computational layers. Training is outside this project scope; the focus is on forward inference with given filter parameters.

## Motivation

Convolutional Neural Networks have revolutionized computer vision by outperforming more complicated object recognition algorithms through simple cascades of image processing operations. This project sheds light on the fundamental operations of CNNs and how they process images layer by layer.

## Features

- Implementation of core CNN building blocks: image normalization, convolution, ReLU, maxpool, fully connected layer, and softmax.
- Multi-layer chaining applying these operations to an input image for classification.
- Use of CIFAR-10 dataset images and pre-trained filter parameters.
- Evaluation of model output accuracy and confusion matrix analysis.
- Debugging support with test inputs and expected intermediate outputs.
- Demo routines to run and visualize results.

## CNN Architecture

- 18 layers total comprising normalization, convolution-ReLU pairs, maxpooling, fully connected layer, and softmax.
- Input: 32x32x3 RGB image.
- Output: 1x1x10 vector with class probabilities.

Refer to the project documentation or source for detailed layer types and sizes.

## Dataset

Uses the CIFAR-10 dataset containing 32x32 color images categorized into 10 classes such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Usage

1. Load the provided datasets:
   - `cifar10testdata.mat` for test images and labels.
   - `CNNparameters.mat` containing filter banks and biases.

2. Implement or use provided layer functions:
   - `applyimnormalize.m`
   - `applyconvolve.m`
   - `applyrelu.m`
   - `applymaxpool.m`
   - `applyfullconnect.m`
   - `applysoftmax.m`

3. Create a main script to sequentially apply each CNN layer function using the supplied parameters on an input image.

4. Run the main script on test images to produce classification probabilities.

5. Optionally run evaluation scripts to compute classification accuracy and confusion matrices over the test set.

## Installation

Requires Matlab with the Image Processing Toolbox (for `imfilter` function).

No additional toolboxes (Computer Vision, Neural Network, or Deep Learning toolbox) or third-party CNN libraries are permitted.

## Evaluation

- Verify correctness of each layer function with provided debugging test data (`debuggingTest.mat`).
- Compute classification accuracy and confusion matrices using the CIFAR-10 test set.
- Analyze which classes are well-classified and which are commonly confused.
- Optionally, test the CNN on new external images resized to 32x32x3.


