from skimage import io
import glob
import os

# Set file
f = open('data_size.txt','a')

# Get file list
list = glob.glob("./DATA/*")

for i in list:

    # Read image
    image = io.imread(i)
    line = i +"\t" + str(image.shape) +"\n"
    # import pdb; pdb.set_trace()

    # Display name and size
    # print(line)
    # import pdb; pdb.set_trace()

    # Write name and size
    f.write(line)

# close file
f.close()
