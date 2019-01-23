class Hinge_loss(Objective):
    
    def __init__(self):
        self.lamb = 0.1
    
    def cal_full_grad(self, X, w):
        N, feature = X.shape
        total = 0;
        for i in range(N):
            total += self.cal_gradient(X[i,:], w, y_from_label_train(i))
        total = total/N
        return total
        
        
    # x: array size 1 by N
    # w: array size 1 by N
    def cal_gradient(self, x, w, y):
        #if x.shape[0] != 1:
        #   x = np.expand_dims(x, axis=0)
        if (2*y - 1) * x.dot(w.T) < 1:
            return self.lamb * w - (2*y - 1)*x
        else:
            return self.lamb * w


    def test_error(self, w_mean):
        loss = 0
        for ind in range(0, test_labels.shape[1]):
            x_it = np.reshape(test_imgs[0,ind,:,:], -1)/255
            x_it = np.array(x_it)
            y_it = 2 * y_from_label_test(ind) - 1
            loss += max(0, 1 - y_it * x_it.dot(w_mean.T))

        loss = loss/test_labels.shape[1]
        loss += self.lamb * np.square(np.linalg.norm(w_mean)) / 2
        return loss

    def train_error(self, w_mean):
        loss = 0
        for ind in range(0, train_labels.shape[1]):
            x_it = np.reshape(train_imgs[0,ind,:,:], -1)/255
            y_it = 2 * y_from_label_train(ind) - 1
            loss += max(0, 1 - y_it * x_it.dot(w_mean.T))

        loss = loss/train_labels.shape[1]
        loss += self.lamb * np.square(np.linalg.norm(w_mean)) / 2
        return loss

    def cal_test_acc(self, w_mean):
        acc = 0
        for ind in range(0, test_labels.shape[1]):
            x_it = np.reshape(test_imgs[0,ind,:,:], -1)/255
            x_it = np.array(x_it)
            y_grd = 2 * y_from_label_test(ind) - 1
            if x_it.dot(w_mean.T) > 0:
                y_pre = 1
            else:
                y_pre = -1
            
            if(y_grd == y_pre):
                acc += 1.0

        acc = acc/test_labels.shape[1]
        return acc