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
from detect_template import score_map

import time
# import pdb; pdb.set_trace()

"""
Calucrate SSD
"""
def choose_template(templates, template_files_index, template_files_all):

    # Define score maps
    score_maps = np.zeros(shape = (len(templates), len(templates)))

    start = time.time()
    print(score_maps.shape)

    # Compute SSD for each template
    for y, i in enumerate(templates):
        for x, j in enumerate(templates):
            score_maps[y, x] = score_map(i, j)

            print(str(y) + ":" + str(x) + ":" + str(time.time() - start))

    # Get best template index
    result = template_files_all[template_files_index[np.argmin(np.sum(score_maps, axis = 0))]]


    return result


"""
Main Process
"""
def main():

    # Open train file
    select_template = configparser.ConfigParser()
    select_template.read('./select_template.conf', 'UTF-8')
    tf = open(select_template.get('files', 'TRAIN'))
    lines = tf.readlines()
    tf.close()
    # import pdb; pdb.set_trace()

    # Get template labels
    template_labels = select_template.get('templates', 'LABELS').split(",")
    # import pdb; pdb.set_trace()

    # Open result file
    rf = open(select_template.get('files', 'RESULT'),'a')

    for s in range(len(template_labels)):

        """
        Read template image
        """
        # Get template file
        template_files_all = [i.replace("\t" + template_labels[s] + "\n", "") for i in lines if i.find(template_labels[s]) >= 0]

        start = time.time()

        # Read template image as gray scale
        # Delete the image size is different, template size must be (2448, 3264)
        templates = []
        template_files_index = []
        for i, v in enumerate(template_files_all):
            if io.imread(v, as_grey = True).shape == (2448, 3264):
                templates.append(io.imread(v, as_grey = True))
                template_files_index.append(i)

                print(str(len(templates)) + ":" + str(time.time() - start))

        """
        Select best template
        """
        template = choose_template(templates, template_files_index, template_files_all)
        # import pdb; pdb.set_trace()

        """
        Write result file
        """
        # Write result
        line = template_labels[s] + " = " + template + "\n"
        rf.write(line)
        # import pdb; pdb.set_trace()

    # Close file
    rf.close()


"""
This script is not executed when called from outside
"""
if __name__ == "__main__":
    main()
