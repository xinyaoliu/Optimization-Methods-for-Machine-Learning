'''
functions that convert labels to probability 1 and 0
'''
def y_from_label_train(ind):
    if dtset=='MNIST':
        if (train_mnist_labels[train_labels[0,ind]] == 6):
            y = 1
        else:
            y = 0
        return y
    elif dtset == 'CIFAR10':
        if train_labels[0,ind] == 1:
            y = 1
        else:
            y = 0
        return y

def y_from_label_test(ind):
    if dtset=='MNIST':
        if(test_mnist_labels[test_labels[0,ind]] == 6):
            y = 1
        else:
            y = 0
        return y

    elif dtset == 'CIFAR10':
        if test_labels[0,ind] == 1:
            y = 1
        else:
            y = 0
        return y
    
# x: array size 1 by N
# w: array size 1 by N
def cal_sigmoid(x, w):
    return expit(np.dot(x, w.T)) # size: 1 by 1
