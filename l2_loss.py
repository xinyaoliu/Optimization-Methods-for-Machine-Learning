'''
Class of L2 loss:
objective function: f(x) = || y - sigmoid(wx) || ^2 
gradient: g(x) =  2*( y - sigmoid(wx) )*{x * (1-sigmoid(wx)) * sigmoid(wx)} 
Image value is normalized to [0, 1] by dividing maximum value of 255
'''
class L2_loss(Objective):
    def cal_full_grad(self, X, w):
        N, feature = X.shape
        total = 0;
        for i in range(N):
            total += self.cal_gradient(X[i,:], w, y_from_label_train(i))
        total = total/N
        return total
        
    def cal_gradient(self, x, w, y):
        z = cal_sigmoid(x, w)
        return 2 * (y - z) * -x * z * (1 - z)
        
    def test_error(self, w_mean):
        
        loss = 0
        for ind in range(0, test_labels.shape[1]):
            x_it = np.reshape(test_imgs[0,ind,:,:], -1)/255
            x_it = np.array(x_it)
            loss += np.square(y_from_label_test(ind) - cal_sigmoid(x_it, w_mean))
        loss = loss/test_labels.shape[1]
        return loss

        raise NotImplementedError
        
    def train_error(self, w_mean):
        loss = 0
        for ind in range(0, train_labels.shape[1]):
            x_it = np.reshape(train_imgs[0,ind,:,:], -1)/255
            loss += np.square(y_from_label_train(ind) - cal_sigmoid(x_it, w_mean))

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