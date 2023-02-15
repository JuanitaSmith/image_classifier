import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torchvision import datasets, transforms, models


def dataloader(data_dir, batch_size, picture_resize, crop_size, std, mean):
    """
    Load images and apply transformations using the torchvision transformation and dataloader library

    Images are organized into train, validation and test datasets.
    Training dataset is randomly augmented

    Args:
        (str) data_dir - main folder where images are stored
        (str) batch_size - number of images that is processed at a time
        (str) picture_resize - resize picture
        (str) crop_size = crop the centre of the image
        (str) std - normalize image colors to standard deviation the model expects
        (str) mean - normalize image colors to mean the model expects
    Returns:
        train_loader - PyTorch tensor to load training data
        validation_loader - PyTorch tensor to load validation data
        test_loader - PyTorch tensor to load test data
        class_to_idx - return tensor to image folder index conversion table
    """

    print("\nInitializing Datasets and Dataloaders...")

    # set transform parameters
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(picture_resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(picture_resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # Load the image datasets with ImageFolder
    data = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
            for x in data_transforms.keys()}

    # return class indexes to save into the model later
    class_to_idx = data['train'].class_to_idx

    # Using the image datasets and the transform definitions, define the dataloaders
    train_loader = torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data['test'], batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader, class_to_idx


def get_class_names(path):
    """
    Load flower name descriptions. A file called 'cat_to_name.json' should be loaded in the root directory

    Returns:
        class_names - names of the flowers
        num_classes - the total number of available flower types
    """

    with open(path, 'r') as f:
        cat_to_name = json.load(f)

    # set number of classes
    num_flowers = len(cat_to_name)

    return cat_to_name, num_flowers


def network(model_name, num_classes, args):
    """ Copy a pre-trained VGG model from PyTorch and only reconfigure the last FC level

        Reference used:
        https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

    Args:
        (str) model_name - PyTorch model to load, only option 'vgg' available but can be extended
        (str) num_classes - number of classes the last FC layer output will be set to
        (dict) args - input parameters containing model training hyper parameters
    Returns:
        model_ft - new model network with last FC layer changed
    """

    model_ft = None

    # define loss function
    criterion = nn.CrossEntropyLoss()

    if model_name == "vgg11_bn":
        """ PyTorch model VGG11_bn """

        model_ft = models.vgg11_bn(pretrained=True)

        # Freeze parameters for the convolutional layers, as this does not need to be retrained
        for param in model_ft.parameters():
            param.requires_grad = False

        # redefine the last step of the classifier to output only 102 classes (types of flowers)
        num_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_features, num_classes)

        # Define optimizer
        optimizer = optim.SGD(model_ft.classifier[6].parameters(), lr=args.learning_rate, momentum=args.momentum)

    elif model_name == 'resnet18':
        model_ft = models.resnet18(pretrained=True)

        # Freeze parameters for the convolutional layers, as this does not need to be retrained
        for param in model_ft.parameters():
            param.requires_grad = False

        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, num_classes)

        # Define optimizer
        optimizer = optim.SGD(model_ft.fc.parameters(), lr=args.learning_rate, momentum=args.momentum)

    print_model(model_ft)

    return model_ft, criterion, optimizer, num_features


def print_model(model_ft):
    """ Print model architecture and parameters that will be trained
    """
    print("\nModel Network Architecture")
    print(model_ft)
    # Double check which parameters are set to be retrained, should only be the last FC level of the model network
    print("\nParams to train:")
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)


def train(model, train_loader, validation_loader, criterion, optimizer, epochs=2, device='cpu'):
    """ Train the model

    Args:
        (str) model - model to be used to train
        (DataLoader) train_loader - PyTorch dataloader to load training data
        (DataLoader) validation_loader - PyTorch dataloader to load validation data
        criterion - loss function to use
        (optim) optimizer - optimizer containing parameters to retrain
        (str) epochs - number of training cycles
        (str) device - gpa setting - cpu or cuda
    Returns:
        model - trained PyTorch model
        optimizer - trained optimizer
        train_losses - losses per epoch
        validation_losses - losses per epoch
    """

    print('\nInitiating Training....')

    # with active_session():

    train_losses, validation_losses = [], []
    for e in range(epochs):

        # Model in training mode, dropout is on, gradient descent calculations is on
        model.train()

        running_loss = 0
        for images, labels in train_loader:

            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            validation_loss = 0
            accuracy = 0

            # Model in inference mode, no dropout
            model.eval()

            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    loss = criterion(output, labels)

                    validation_loss += loss.item()

                    # Calculate accuracy
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss / len(train_loader))
            validation_losses.append(validation_loss / len(validation_loader))

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
                  "Validation Loss: {:.3f}.. ".format(validation_loss / len(validation_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy / len(validation_loader)))

    return model, optimizer, train_losses, validation_losses


def test(model, test_loader, criterion, device='cpu'):

    print('\nStarting test accuracy calculation.....')

    test_loss = 0
    accuracy = 0

    # Model in inference mode, no dropout
    model.eval()

    test_losses = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            loss = criterion(output, labels)
            test_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        test_losses.append(test_loss / len(test_loader))
        epoch_accuracy = round(accuracy / len(test_loader),3)

        print("\nTest Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
              "Test Accuracy: {:.3f}".format(epoch_accuracy))

    return epoch_accuracy


def save_checkpoint(path, input_size, output_size, batch_size, epochs, class_to_idx, model_name, model,
                    optimizer, learning_rate, momentum):

    print('\nSaving checkpoint to location...', path)

    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'batch_size': batch_size,
                  'epochs': epochs,
                  'class_to_idx': class_to_idx,
                  'model_name': model_name,
                  'model': model,
                  'model_state_dict': model.state_dict(),
                  'optimizer': optimizer,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'learning_rate': learning_rate,
                  'momentum': momentum}

    torch.save(checkpoint, path)


def load_checkpoint(filepath, device):
    print('\nLoading checkpoint from path', filepath)

    checkpoint = torch.load(filepath, map_location=device)

    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print_model(model)

    return model, optimizer, checkpoint


def process_image(image, picture_resize, crop_size, mean, std):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model

    Reference:
      https://towardsdatascience.com/a-beginners-tutorial-on-building-an-ai-image-classifier-using-pytorch-6f85cb69cba7

    Returns:
        tensor_image - Transformed PIL image converted into a tensor
    """

    # Resize the picture
    size = (picture_resize, picture_resize)
    image.thumbnail(size)

    # Crop picture to same size as PyTorch tensors in dataloaders
    width, height = image.size
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    image = image.crop((left, top, right, bottom))

    # change colours from range 0-255 to 0-1
    np_image = np.array(image) / 256

    # Normalize based on the preset mean and standard deviation
    np_image = (np_image - mean) / std

    # color channel should be first column, then height and width
    np_image = np.transpose(np_image, (2, 0, 1))

    # turn image into a tensor object
    tensor_image = torch.from_numpy(np_image)
    tensor_image = tensor_image.float()

    return tensor_image

def select_image(image_path, class_names):
    """
    Randomly select and image from the test folder

    Args:
    (str) image_path - folder where the testing data is stored
    (str) class_names - names of the flowers

    Returns:
    img - PIL image randomly selected and opened, without any transformations
    name - actual name of the flower for validation purposes
    """

    class_selection = random.choices(os.listdir(image_path), k=1)[0]
    name = class_names[str(class_selection)]
    image_class_path = image_path + class_selection + '/'
    image_full_path = image_class_path + random.choices(os.listdir(image_class_path), k=1)[0]
    img = Image.open(image_full_path)
    return img, name

def predict(image, model, mean, std, topk=5, picture_resize=256, crop_size=224, device='cpu'):
    """
    Predict the class (or classes) of an image using a trained deep learning model

    Args:
    (object) image: PIL image
    (object) model: model architecture, transfer learning from PyTorch with adjusted classifier
    (list) mean: mean used picture color normalization in ImageNet used for this model
    (list) std: mean used picture color normalization in ImageNet used for this model
    (int)) top_k: number of top classes to predict
    (int)) picture_resize: resize images to be the same size
    (int)) crop_size: crop the middle of the image to this size

    Returns:
    probs: probability score of the top predicted classes
    classes: top predicted classes
    """

    # set model into evaluation mode
    model.eval()

    # convert image into a tensor ready for input into the model
    tensor_image = process_image(image, picture_resize, crop_size, mean, std)
    # Add a fourth dimension to the beginning to indicate batch size
    tensor_image = tensor_image[np.newaxis, :]

    # move image to chosen device
    tensor_image = tensor_image.to(device)

    # Calculate the class probabilities for the image
    with torch.no_grad():
        output = model.forward(tensor_image)

    sm = torch.nn.Softmax(dim=1)
    ps = sm(output)
    probs, classes = ps.topk(topk, dim=1)

    return probs, classes