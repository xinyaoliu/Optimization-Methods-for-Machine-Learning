'''
Load and Preprocess Dataset
We download the original dataset from the official public website of MNIST and CFAR-10
Binary classificadtion labels of MNIST are number '6' and '8', corresponding to 1 and 0; labels of CFAR10 are type
'cat' and 'dog', corresponding to 1 and 0.
Images are all resized to 1 by N array, which is easier for further processing
'''
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
dtset = 'CIFAR10'

if dtset == 'MNIST':
    # y = 1 if digit = 6 
    # y = 0 if digit = 8
    train_mnist_labels = read_idx('data/mnist/raw/train-labels-idx1-ubyte')
    train_mnist_imgs = read_idx('data/mnist/raw/train-images-idx3-ubyte')

    test_mnist_labels = read_idx('data/mnist/raw/t10k-labels-idx1-ubyte')
    test_mnist_imgs = read_idx('data/mnist/raw/t10k-images-idx3-ubyte')

    # Pickout the labeled training dataset
    train_labels = np.array(np.where((train_mnist_labels == 6) | (train_mnist_labels == 8)))
    train_imgs = train_mnist_imgs[train_labels]

    # Pickout the labeled test dataset
    test_labels = np.array(np.where((test_mnist_labels == 6) | (test_mnist_labels == 8)))
    test_imgs = test_mnist_imgs[test_labels]

elif dtset == 'CIFAR10':
    # y = 1 if lebel = cat
    # y = 0 if label = dog
    
    CAT1 = 1
    CAT2 = 5
    input_size = 32 * 32
    #CIFAR dataset
    train_dataset = dataset.CIFAR10(root = './data/cifar', 
                               train = True, 
                               transform = transforms.ToTensor(), 
                               download = True)
    test_dataset = dataset.CIFAR10(root = './data/cifar', 
                               train = False, 
                               transform = transforms.ToTensor(), 
                               download = True)

    official_train_images = np.array(train_dataset.train_data).astype(np.float)
    official_train_labels = np.array(train_dataset.train_labels).astype(np.int)

    official_test_images = np.array(test_dataset.test_data).astype(np.float)
    official_test_labels = np.array(test_dataset.test_labels).astype(np.int)

    mask_cat_dog = (official_train_labels == CAT1 ) | (official_train_labels == CAT2)
    images_cat_dog = official_train_images[mask_cat_dog]
    labels_cat_dog = (official_train_labels[mask_cat_dog] == CAT1 ).astype(np.int)
    
    train_labels = labels_cat_dog.reshape(1,-1)
    train_imgs = images_cat_dog
    train_imgs = np.expand_dims(np.mean(train_imgs, axis=3), axis=0)

    mask_cat_dog_t = (official_test_labels == CAT1 ) | (official_test_labels == CAT2)
    images_cat_dog_t = official_test_images[mask_cat_dog_t]
    labels_cat_dog_t = (official_test_labels[mask_cat_dog_t] == CAT1 ).astype(np.int)

    # Pickout the labeled test dataset
    test_labels = labels_cat_dog_t.reshape(1,-1)
    test_imgs = images_cat_dog_t
    test_imgs = np.expand_dims(np.mean(test_imgs, axis=3), axis=0)


# sample_size = train_labels.shape[1]