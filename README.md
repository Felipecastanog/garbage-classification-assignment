# garbage-classification-assignment

This project implements a garbage classification model using PyTorch to categorize waste into "green," "blue," "black," and "other" categories. Classification is based on a combination of images and text descriptions, helping to identify the correct category for each waste item. This system was developed as part of **ENEL 645** at the University of Calgary to assist with waste management in the city.

Name: Felipe Castano 

## Project Description

This project aims to solve the problem of garbage classification by combining image processing and natural language understanding to interpret user-provided waste descriptions. We use a Convolutional Neural Network (ResNet18) for image analysis and BERT for text analysis. Outputs from both models are combined and passed through fully connected layers for final classification. This solution helps residents correctly categorize their waste, contributing to environmental sustainability.

## Installing Dependencies

Install the required dependencies in the requirements.txt file

### Key Requirements

- `torch`: Deep learning framework for building and training the model.
- `torchvision`: Contains pretrained models and image transformation tools.
- `transformers`: NLP models library, used here to load BERT.
- `sklearn`: Metric and evaluation tools.
- `Pillow`: Image manipulation.
- `matplotlib`: Data visualization and graphing.

## Usage

### Training and Testing the Combined Model

This code integrates both image and text data to classify garbage items into the categories "green," "blue," "black," and "other." Using a combined model, it processes images with a ResNet18 architecture and text descriptions with a BERT model, training both together to optimize classification accuracy.

The code structure includes:
- **Data Preparation**: The `GarbageDataset` class loads images and corresponding text descriptions, organizes them into tensors for PyTorch, and applies necessary transformations.
- **Model Architecture**: The `EnhancedCombinedModel` combines features extracted from ResNet18 (for images) and BERT (for text) through additional fully connected layers, batch normalization, and dropout, ensuring stable and accurate classification.
- **Training and Validation**: The `train_model` function trains the combined model on the dataset and evaluates it using validation data at each epoch.
- **Evaluation**: The `evaluate_model` function evaluates model performance on the test set, providing accuracy and a confusion matrix.

### Data Access

As this project was developed in an academic setting, please consult with our instructor to access the specific data.

## Model Design Decisions

- **Convolutional Neural Network (ResNet18)**: Used to process image features and extract relevant visual information.
- **BERT Model**: Employed to capture the semantics of text descriptions, identifying keywords like material or type of waste.
- **Combined Architecture**: ResNet and BERT outputs are combined in fully connected layers with batch normalization and dropout to improve stability and prevent overfitting.
- **Optimization**: We applied hyperparameter tuning techniques like adjusted learning rates and batch sizes to achieve an accuracy above 80%.
