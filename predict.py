import argparse
import sys

import numpy as np

import image_utilities
import model_utilities

# Global parameters
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
PICTURE_RESIZE = 256
CROP_SIZE = 224
DATA_DIR = 'flowers/test/'


def get_parser():
    """
    Get user input parameters that was entered at command line and validate them

    :return:
    args (dict) : parameter and value pairs
    """

    parser = argparse.ArgumentParser(description='Imagine Recognition Prediction')
    parser.add_argument("path_to_image", default=DATA_DIR, type=str,
                        help="Directory where test images are stored that the model has not been trained on")
    parser.add_argument("save_dir", default='checkpoint.pth', type=str,
                        help="Path to saved checkpoint")
    parser.add_argument("--top_k", default=5, type=int,
                        help="top K most likely classes to return")
    parser.add_argument("--category_names", default="cat_to_name.json", type=str,
                        help="Mapping of categories to real names")
    parser.add_argument("--gpu", default='cpu', type=str, choices=('cuda', 'cpu'),
                        help="Select to run model on GPA - options 'cuda' or 'cpu'")

    args = parser.parse_args()
    if not args.save_dir[-4:] == '.pth':
        sys.exit('Invalid checkpoint file, should end with .pth')

    if not args.category_names[-5:] == '.json':
        sys.exit('Invalid category names path, should end with .json')

    return args


if __name__ == "__main__":
    # get user input parameters and validate them
    args = get_parser()
    print('\nPrediction parameters chosen:', args)

    # Get class (flower) names
    class_names, num_classes = model_utilities.get_class_names(args.category_names)

    model, optimizer, checkpoint = model_utilities.load_checkpoint(args.save_dir, args.gpu)

    # Send model to selected device
    model = model.to(args.gpu)

     # select a random image from the image test path provided
    image, actual_class_name = model_utilities.select_image(args.path_to_image, class_names)

    # predict the top K classes for the image
    probs, classes = model_utilities.predict(image=image,
                                             model=model,
                                             mean=MEAN,
                                             std=STD,
                                             topk=args.top_k,
                                             picture_resize=PICTURE_RESIZE,
                                             crop_size=CROP_SIZE,
                                             device=args.gpu)

    # visualize results
    image_utilities.view_classify(image, probs, classes, actual_class_name, class_names,
                                  checkpoint['class_to_idx'])
