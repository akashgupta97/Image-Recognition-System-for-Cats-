
# # Image Recognition System

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

get_ipython().magic('matplotlib inline')


# ## 2 - Overview of the Problem set ##
#
# **Problem Statement**:  given a dataset ("data.h5") containing:
#     - a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
#     - a test set of m_test images labeled as cat or non-cat
#     - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). Thus, each image is square (height = num_px) and (width = num_px).
#
#  build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.
#
# Let's get more familiar with the dataset. Load the data by running the following code.

# In[32]:

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# We added "_orig" at the end of image datasets (train and test) because we are going to preprocess them. After preprocessing, we will end up with train_set_x and test_set_x (the labels train_set_y and test_set_y don't need any preprocessing).
#
# Each line of your train_set_x_orig and test_set_x_orig is an array representing an image. You can visualize an example by running the following code. Feel free also to change the `index` value and re-run to see other images.

# In[33]:

# Example of a picture
index = 27
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
    "utf-8") + "' picture.")


### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))

# **Expected Output for m_train, m_test and num_px**:
# <table style="width:15%">
#   <tr>
#     <td>**m_train**</td>
#     <td> 209 </td>
#   </tr>
#
#   <tr>
#     <td>**m_test**</td>
#     <td> 50 </td>
#   </tr>
#
#   <tr>
#     <td>**num_px**</td>
#     <td> 64 </td>
#   </tr>
#
# </table>
#
# In[35]:

# Reshape the training and test examples

### START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
### END CODE HERE ###

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

# **Expected Output**:
#
# <table style="width:35%">
#   <tr>
#     <td>**train_set_x_flatten shape**</td>
#     <td> (12288, 209)</td>
#   </tr>
#   <tr>
#     <td>**train_set_y shape**</td>
#     <td>(1, 209)</td>
#   </tr>
#   <tr>
#     <td>**test_set_x_flatten shape**</td>
#     <td>(12288, 50)</td>
#   </tr>
#   <tr>
#     <td>**test_set_y shape**</td>
#     <td>(1, 50)</td>
#   </tr>
#   <tr>
#   <td>**sanity check after reshaping**</td>
#   <td>[17 31 56 22 33]</td>
#   </tr>
# </table>

