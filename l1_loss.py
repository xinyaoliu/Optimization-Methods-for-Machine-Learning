'''
Class of L1 loss:
objective function: f(x) = || y - sigmoid(wx) || 
gradient: g(x) = - {x * (1-sigmoid(wx)) * sigmoid(wx)} * sign(y)
Image value is normalized to [0, 1] by dividing maximum value of 255
'''
class L1_loss(Objective):
    
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
        if y==1:
        #    print("y=1", cal_sigmoid(x,w))
            return -x * cal_sigmoid(x,w) * (1-cal_sigmoid(x,w))# size: 1 by N
        else:
        #    print("y=0", cal_sigmoid(x,w))
            return x * cal_sigmoid(x,w) * (1-cal_sigmoid(x,w)) # size: 1 by N


    def test_error(self, w_mean):
        loss = 0
        for ind in range(0, test_labels.shape[1]):
            x_it = np.reshape(test_imgs[0,ind,:,:], -1)/255
            x_it = np.array(x_it)
            loss += np.abs(y_from_label_test(ind) - cal_sigmoid(x_it, w_mean))

        loss = loss/test_labels.shape[1]
        return loss

    def train_error(self, w_mean):
        loss = 0
        for ind in range(0, train_labels.shape[1]):
            x_it = np.reshape(train_imgs[0,ind,:,:], -1)/255
            loss += np.abs(y_from_label_train(ind) - cal_sigmoid(x_it, w_mean))

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