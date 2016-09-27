
images = loadMNISTImages('../Data/train-images.idx3-ubyte');
labels = loadMNISTLabels('../Data/train-labels.idx1-ubyte');

images_test = loadMNISTImages('../Data/t10k-images.idx3-ubyte');
labels_test = loadMNISTLabels('../Data/t10k-labels.idx1-ubyte');

[W1, W2] = BackProp(labels, images', labels_test, images_test');
