from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_examples = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_examples):
      scores = X[i].dot(W) # (1, C)
      scores -= np.max(scores)
      softmax = np.exp(scores)/np.sum(np.exp(scores))
      loss -= np.log(softmax[y[i]])
      for k in range(num_classes):
        if k == y[i]:
          indicator = 1
        else:
          indicator = 0
        dW[:, k] += (softmax[k] - indicator)*X[i]

    loss /= num_examples
    dW /= num_examples

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W) # (D, C)
    num_examples = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X @ W # (N, C)
    scores -= np.max(scores, axis = 1).reshape(-1,1)
    softmax = np.exp(scores)/np.sum(np.exp(scores), axis = 1).reshape(-1,1)
    loss += np.sum(-np.log(softmax[np.arange(y.shape[0]), y]))
    mask = np.zeros_like(softmax)
    mask[np.arange(y.shape[0]), y] += 1
    dW = X.T @ (softmax - mask)

    loss /= num_examples
    dW /= num_examples

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
