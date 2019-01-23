'''
Class of Logistic loss:
objective function: f(x) =  y * log(sigmoid(wx)) + (1-y) * log(1-sigmoid(wx)) 
gradient: g(x) = y * sigmoid(-wx) * x + (1 - y) * sigmoid(wx) * (-x) 
Image value is normalized to [0, 1] by dividing maximum value of 255
'''
class Logistic_loss(Objective):
    
    def cal_full_grad(self, X, w):
        N, feature = X.shape
        total = 0;
        for i in range(N):
            total += self.cal_gradient(X[i,:], w, y_from_label_train(i))
        total = total/N
        return total
    
    
    def cal_gradient(self, X, w, yy):
        wx = X.dot(w.T)
        logistn = expit(-wx)
        logist = 1-logistn
        updatevec = yy*logistn + (yy-1)*logist
        update = X*updatevec
        return -update
    
    def cal_hessian(self, X, w):
        z = cal_sigmoid(X, w)
        return z * (1-z) * np.matmul(X.T, X)


    def get_loss(self, X, w, y):
        wx = X.dot(w.T)
        logist = expit(wx)
        if y == 1:
            return -np.log(np.clip(logist,1e-7,1-1e-7))
        else:
            return -np.log(1-np.clip(logist,1e-7,1-1e-7))


    def test_error(self, w_mean):
        loss = 0
        for ind in range(0, test_labels.shape[1]):
            x_it = np.reshape(test_imgs[0,ind,:,:], -1)/255
            x_it = np.array(x_it)
            loss += self.get_loss(x_it, w_mean, y_from_label_test(ind))

        loss = loss/test_labels.shape[1]
        return loss

    def train_error(self, w_mean):
        loss = 0
        for ind in range(0, train_labels.shape[1]):
            x_it = np.reshape(train_imgs[0,ind,:,:], -1)/255
            loss += self.get_loss(x_it, w_mean, y_from_label_train(ind))

        loss = loss/train_labels.shape[1]
        return loss

    def cal_test_acc(self, w_mean):
        acc = 0
        for ind in range(0, test_labels.shape[1]):
            x_it = np.reshape(test_imgs[0,ind,:,:], -1)/255
            x_it = np.array(x_it)
            y_grd = y_from_label_test(ind)
            if(cal_sigmoid(x_it, w_mean) > 0.5):
                y_pre = 1
            else:
                y_pre = 0

            if(y_grd == y_pre):
                acc += 1.0

        acc = acc/test_labels.shape[1]
        return acc