"""
Load module
"""
import sys
import os
import glob
import configparser
import numpy as np
from skimage import io
from skimage.transform import rescale
import matplotlib.pyplot as plt
import matplotlib.cm as cm


"""
Calculate SSD and return score map
"""
def score_map(template, target):

    """
    Calculate SSD
    """
    # Get template hight and width
    th = template.shape[0]
    tw = template.shape[1]
    # import pdb; pdb.set_trace()

    # Varience for  SSD at each position, fill with zero at first
    score_map = np.zeros(shape = (target.shape[0] - th + 1,
                                  target.shape[1] - tw + 1))
    # import pdb; pdb.set_trace()

    # Calculate SSD
    for y in range(score_map.shape[0]):
        for x in range(score_map.shape[1]):

            # Get sum of squared differences
            diff = target[y:y + th, x:x + tw] - template
            # import pdb; pdb.set_trace()
            score_map[y,x] = np.square(diff).sum()
            # import pdb; pdb.set_trace()

    return score_map


"""
Detect template and return template index
"""
def get_minimum_score_index(target, templates):

    """
    Calculate SSD each template and get score maps
    """
    # Set score map list
    score_maps = []

    for template in templates:

        # Get score map for each template
        score_maps.append(score_map(template, target))
        # import pdb; pdb.set_trace()

    # Get minimum flattened index with agrgmin()
    result = np.argmin(score_maps)
    # import pdb; pdb.set_trace()

    return result


"""
Visualize result
"""
def visualize_result(target, target_file, templates,
                     template_files,template_files_index ,result):
    # Difine window size and elements
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (8, 4))

    # Define ax1 and display template image in grayscale
    # Do not show axes and set title
    ax1.imshow(target, cmap = cm.Greys_r)
    ax1.set_axis_off()
    ax1.set_title('target\n'
                  + target_file)

    # Define ax2 and display template image in grayscale
    # Do not show axes and set title
    ax2.imshow(templates[result], cmap = cm.Greys_r)
    ax2.set_axis_off()
    ax2.set_title('template\n'
                  + template_files[result]
                  + '\n'
                  + template_files_index[result])

    # Display
    plt.show()
    # import pdb; pdb.set_trace()


"""
Main Process
"""
def main():
    """
    Read template image
    """
    # Get image path
    detect_template = configparser.ConfigParser()
    detect_template.read('./detect_template.conf', 'UTF-8')
    template_files_index = [i.upper() for i in detect_template['templates']]
    template_files = [detect_template.get('templates', i) for i in template_files_index]
    # import pdb; pdb.set_trace()

    # Read image as gray scale
    templates = [io.imread(i, as_grey = True) for i in template_files]
    # import pdb; pdb.set_trace()

    # Check template size, templates must be (2448, 3264)
    """
    for i,j in enumerate(templates):
        print(i, j.shape)
    import pdb; pdb.set_trace()
    """
    # Get target file list
    target_files = glob.glob(detect_template.get('targets', 'DIR'))
    # import pdb; pdb.set_trace()

    # Open train file
    tf = open(detect_template.get('files', 'TRAIN'))
    lines = tf.readlines()
    tf.close()
    # import pdb; pdb.set_trace()

    # Get igenore labels
    ignore_labels = detect_template.get('ignores', 'LABELS').split(",")
    # import pdb; pdb.set_trace()

    # Get ignore files
    ignore_files = []
    for i in lines:
        # Append to the list if the ignore label is present
        for j in ignore_labels:
            if i.find(j) >= 0:
                ignore_files.append(i.replace("\t" + j + "\n", ""))
    # import pdb; pdb.set_trace()

    # Exclude ignore files from target files
    target_files = list(set(target_files) ^ set(ignore_files))
    # import pdb; pdb.set_trace()

    # Open result file
    rf = open(detect_template.get('files', 'RESULT'),'w')

    # Set correct counter
    n = 0

    # Set break counter
    m = 0

    for i in target_files:

        # Read image as gray scale
        target = io.imread(i, as_grey = True)
        # import pdb; pdb.set_trace()

        # Check target size, must be same template (2448, 3264)
        if target.shape != (2448, 3264):
            m = m + 1
            continue

        # Detect template
        result = get_minimum_score_index(target, templates)
        # import pdb; pdb.set_trace()

        # Show result
        # visualize_result(target, i, templates, template_files,template_files_index ,result)
        # import pdb; pdb.set_trace()

        # Write result to file
        line = i + "\t" + str(template_files_index[result]) +"\n"
        rf.write(line)

        # Find result in train data
        for k in lines:

            # If result is a correct, incremant correct counter
            if k == line:
                n = n + 1
                # import pdb; pdb.set_trace()

    # Write summary
    line = "Correct: " + str(n) + "\n" + "Total: " + str(len(lines) - len(ignore_files) - m) + "\n" + "Accuracy: " + str(n / (len(lines) - len(ignore_files) - m))
    rf.write(line)

    # Close file
    rf.close()



"""
This script is not executed when called from outside
"""
if __name__ == "__main__":
    main()
