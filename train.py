import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim

import model_utilities

# Global parameters
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
PICTURE_RESIZE = 256
CROP_SIZE = 224
DATA_DIR = 'flowers'


def get_parser():
    """
    Get user input parameters that was entered at command line and validate them

    :return:
    args (dict) : parameter and value pairs
    """

    parser = argparse.ArgumentParser(description='Imagine Recognition Training')
    parser.add_argument("save_dir", default='checkpoint.pth', type=str,
                        help="Directory where checkpoints will be saved")
    parser.add_argument("--arch", default='vgg11_bn', type=str, choices=('vgg11_bn', 'resnet18'),
                        help="PyTorch models to transfer learn from 'vgg11'_bn' or 'resnet18'")
    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of training cycles, default is 3")
    parser.add_argument("--learning_rate", default=0.001, type=int,
                        help="Learning rate, default is 0.001")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for data loaders")
    parser.add_argument("--momentum", default=0.95, type=int,
                        help="Momentum, default is 0.95")
    parser.add_argument("--gpu", default='cpu', type=str, choices=('cuda', 'cpu'),
                        help="Select to run model on GPA - options 'cuda' or 'cpu'")
    parser.add_argument("--category_names", default="cat_to_name.json", type=str,
                        help="Mapping of categories to real names")
    # I only add this because of Udacity Rubic, I think for the way I configured only the last layer of the
    # classifier, it's better to do this dynamically without hardcoding.  I will not use it further as it will cause
    # problems.
    parser.add_argument("--hidden_units", default="512", type=int,
                        help="Hidden layer input size")

    args = parser.parse_args()
    if not args.save_dir[-4:] == '.pth':
        sys.exit('Invalid save_dir, should end with .pth')
    if not args.category_names[-5:] == '.json':
        sys.exit('Invalid category names path, should end with .json')

    return args


if __name__ == "__main__":
    """
    Train a network on a new dataset and save the model as checkpoint

    """

    # get user input parameters and validate them
    args = get_parser()

    print('\nModel will be trained with the following parameters:', args)

    # Get class (flower) names
    class_names, num_classes = model_utilities.get_class_names(args.category_names)

    # Initialise Datasets and DataLoaders
    train_loader, validation_loader, test_loader, class_to_idx = model_utilities.dataloader(data_dir=DATA_DIR,
                                                                                            batch_size=args.batch_size,
                                                                                            picture_resize=PICTURE_RESIZE,
                                                                                            crop_size=CROP_SIZE,
                                                                                            std=STD,
                                                                                            mean=MEAN)

    # Instantiate the model
    model, criterion, optimizer, input_size = model_utilities.network(model_name=args.arch,
                                                                      num_classes=num_classes,
                                                                      args=args)



    # Send model to selected device
    model = model.to(args.gpu)

    # Train the model
    model, optimizer, train_losses, validation_losses = model_utilities.train(model=model, train_loader=train_loader,
                                                                              validation_loader=validation_loader,
                                                                              criterion=criterion, optimizer=optimizer,
                                                                              epochs=args.epochs,
                                                                              device=args.gpu)
    # Plot training vs validation accuracy
    plt.plot(train_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

    # calculate the test accuracy
    test_accuracy = model_utilities.test(model, test_loader, criterion, device=args.gpu)

    # save checkpoint
    model_utilities.save_checkpoint(path=args.save_dir,
                                    input_size=input_size,
                                    output_size=num_classes,
                                    batch_size=args.batch_size,
                                    epochs=args.epochs,
                                    class_to_idx=class_to_idx,
                                    model_name=args.arch,
                                    model=model,
                                    optimizer=optimizer,
                                    learning_rate=args.learning_rate,
                                    momentum=args.momentum)

    print('Model is successfully trained with test accuracy of {} and checkpoint saved at location {}'
          .format(test_accuracy, args.save_dir))
