import matplotlib.pyplot as plt
import numpy as np


def view_classify(img, probs, classes, actual_class_name, class_names, class_to_idx):
    """
    Function for viewing one image with it's actual name and it's predicted classes.

    Args:
    (PIL) img: original image
    (tensor) probs: top K probabilities that is predicted for an image
    (str) actual_class_name: actual class name of the image
    (dict) class_names: link of image folder to actual name of class
    (dict) class_to_idx: link of class to image image folder
    """

    # get the class names for the topk predicted classes for row labels
    top_class_names = []
    classes = classes.cpu()
    for i, j in enumerate(classes.numpy().squeeze()):
        index = list(class_to_idx.keys())[j.item()]
        top_class_names.append(class_names[str(index)])

    probs = probs.cpu()
    probs = probs.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), nrows=2)
    ax1.set_title(actual_class_name)
    ax1.imshow(img)
    ax1.axis('off')

    ax2.invert_yaxis()
    ax2.barh(np.arange(5), probs)
    ax2.set_yticks(np.arange(5))

    ax2.set_yticklabels(top_class_names, size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.show()
