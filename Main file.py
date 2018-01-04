
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

# In[36]:

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


# <font color='blue'>
# **What you need to remember:**
#
# Common steps for pre-processing a new dataset are:
# - Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
# - Reshape the datasets such that each example is now a vector of size (num_px \* num_px \* 3, 1)
# - "Standardize" the data

# ## 3 - General Architecture of the learning algorithm ##
#
# It's time to design a simple algorithm to distinguish cat images from non-cat images.
#

# build a Logistic Regression, using a Neural Network mindset. The following Figure explains why **Logistic Regression is actually a very simple Neural Network!**
#
# <img src="images/LogReg_kiank.png" style="width:650px;height:400px;">
#
# **Mathematical expression of the algorithm**:
#
# For one example $x^{(i)}$:
# $$z^{(i)} = w^T x^{(i)} + b \tag{1}$$
# $$\hat{y}^{(i)} = a^{(i)} = sigmoid(z^{(i)})\tag{2}$$
# $$ \mathcal{L}(a^{(i)}, y^{(i)}) =  - y^{(i)}  \log(a^{(i)}) - (1-y^{(i)} )  \log(1-a^{(i)})\tag{3}$$
#
# The cost is then computed by summing over all training examples:
# $$ J = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(a^{(i)}, y^{(i)})\tag{6}$$
#

# **Key steps**:
# In this exercise, carry out the following steps:
#     - Initialize the parameters of the model
#     - Learn the parameters for the model by minimizing the cost
#     - Use the learned parameters to make predictions (on the test set)
#     - Analyse the results and conclude

# ## 4 - Building the parts of our algorithm ##
#
# The main steps for building a Neural Network are:
# 1. Define the model structure (such as number of input features)
# 2. Initialize the model's parameters
# 3. Loop:
#     - Calculate current loss (forward propagation)
#     - Calculate current gradient (backward propagation)
#     - Update parameters (gradient descent)
#
# build 1-3 separately and integrate them into one function we call `model()`.
#
# ### 4.1 - Helper functions
#
# **Exercise**: Using your code from "Python Basics", implement `sigmoid()`. As you've seen in the figure above, you need to compute $sigmoid( w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$ to make predictions. Use np.exp().

# In[37]:

# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z
    Arguments:
    z -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###

    return s


# In[38]:

print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))


# **Expected Output**:
#
# <table>
#   <tr>
#     <td>**sigmoid([0, 2])**</td>
#     <td> [ 0.5         0.88079708]</td>
#   </tr>
# </table>

# ### 4.2 - Initializing parameters
#
# **Exercise:** Implement parameter initialization in the cell below. You have to initialize w as a vector of zeros. If you don't know what numpy function to use, look up np.zeros() in the Numpy library's documentation.

# In[39]:

# GRADED FUNCTION: initialize_with_zeros

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    w = np.zeros([dim, 1])
    b = 0
    ### END CODE HERE ###

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


# In[40]:

dim = 2
w, b = initialize_with_zeros(dim)
print("w = " + str(w))
print("b = " + str(b))


# **Expected Output**:
#
#
# <table style="width:15%">
#     <tr>
#         <td>  ** w **  </td>
#         <td> [[ 0.]
#  [ 0.]] </td>
#     </tr>
#     <tr>
#         <td>  ** b **  </td>
#         <td> 0 </td>
#     </tr>
# </table>
#
# For image inputs, w will be of shape (num_px $\times$ num_px $\times$ 3, 1).


# **Hints**:
#
# Forward Propagation:
# - You get X
# - You compute $A = \sigma(w^T X + b) = (a^{(0)}, a^{(1)}, ..., a^{(m-1)}, a^{(m)})$
# - You calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$
#
# Here are the two formulas you will be using:
#
# $$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
# $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$

# In[41]:

# GRADED FUNCTION: propagate

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
"""
    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -1 / m * (np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T))  # compute cost
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = 1 / m * (np.dot(X, (A - Y).T))
    db = 1 / m * (np.sum(A - Y))
    ### END CODE HERE ###

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


# In[42]:

w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))


# **Expected Output**:
#
# <table style="width:50%">
#     <tr>
#         <td>  ** dw **  </td>
#         <td> [[ 0.99993216]
#  [ 1.99980262]]</td>
#     </tr>
#     <tr>
#         <td>  ** db **  </td>
#         <td> 0.499935230625 </td>
#     </tr>
#     <tr>
#         <td>  ** cost **  </td>
#         <td> 6.000064773192205</td>
#     </tr>
#
# </table>


# ### d) Optimization
# - You have initialized your parameters.
# - You are also able to compute a cost function and its gradient.
# - Now, you want to update the parameters using gradient descent.

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


# In[44]:

params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))


# **Expected Output**:
#
# <table style="width:40%">
#     <tr>
#        <td> **w** </td>
#        <td>[[ 0.1124579 ]
#  [ 0.23106775]] </td>
#     </tr>
#
#     <tr>
#        <td> **b** </td>
#        <td> 1.55930492484 </td>
#     </tr>
#     <tr>
#        <td> **dw** </td>
#        <td> [[ 0.90158428]
#  [ 1.76250842]] </td>
#     </tr>
#     <tr>
#        <td> **db** </td>
#        <td> 0.430462071679 </td>
#     </tr>
#
# </table>
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X) + b)
    ### END CODE HERE ###

    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        if (A[0][i] <= 0.5):
            Y_prediction[0][i] = 0
        else:
            Y_prediction[0][i] = 1
            ### END CODE HERE ###

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


# In[46]:

print("predictions = " + str(predict(w, b, X)))


# **Expected Output**:
#
# <table style="width:30%">
#     <tr>
#          <td>
#              **predictions**
#          </td>
#           <td>
#             [[ 1.  1.]]
#          </td>
#    </tr>
#
# </table>
#

# <font color='blue'>
# **What to remember:**
# - Initialize (w,b)
# - Optimize the loss iteratively to learn parameters (w,b):
#     - computing the cost and its gradient
#     - updating the parameters using gradient descent
# - Use the learned (w,b) to predict the labels for a given set of examples

# ## 5 - Merge all functions into a model ##
#
#
# **Exercise:** Implement the model function. Use the following notation:
#     - Y_prediction for your predictions on the test set
#     - Y_prediction_train for your predictions on the train set
#     - w, costs, grads for the outputs of optimize()

# In[49]:
