class Newton(Opti_algo):
    def train(self):
        T = 5000 # number of iterations
        S = train_labels.shape[1]
        test_loss = []
        train_loss = []
        theory_loss = []
        weight_norm = []
        w_tt = np.random.randn(1, train_imgs.shape[2]*train_imgs.shape[3])
        w_mean = np.zeros((1, train_imgs.shape[2]*train_imgs.shape[3]))
        w_tt = np.array(w_tt)
        w_t=[]
        w_t.append(w_tt)
        alpha = 0.001
        dim = w_tt.shape[1]
        g_it = np.zeros((1, dim))
        h_it = np.zeros((dim, dim))
        batch_idx = 0
        test_acc = []
        start_sgd = timeit.default_timer()
        runtime_list = []

        # Start Iteration Loop
        for t in range(0, T):
            if t % 100 == 0:
                print("running", t)
                

            # Select random it
            i_t = np.random.randint(S, size=1)
            # Locate label
            y_it = y_from_label_train(i_t)
            # Pickout x_it
            x_it = np.reshape(train_imgs[0,i_t,:,:], (1, train_imgs.shape[2]*train_imgs.shape[3]))/255
            # Do inner production
            a = self.obj.cal_gradient(x_it, w_tt, y_it)
            g_it += self.obj.cal_gradient(x_it, w_tt, y_it)
            # Calculate Hessian
            h_it += self.obj.cal_hessian(x_it, w_tt) + alpha * np.identity(dim)

            # Do w_t update
            w_tt = w_tt - (np.matmul(np.linalg.inv(h_it), g_it.T).T)


            w_t.append(w_tt)
            w_mean = w_tt

            # Append test error
            
            if t % 100 == 0:
                stop_sgd = timeit.default_timer()
                runtime  = stop_sgd - start_sgd
                print("test accuracy:", self.obj.cal_test_acc(w_mean))
                test_loss.append(self.obj.test_error(w_mean))
                train_loss.append(self.obj.train_error(w_mean))
                test_acc.append(self.obj.cal_test_acc(w_mean))
                runtime_list.append(runtime)
                weight_norm.append(w_mean)
                
        np.save(str(dtset)+'newton_logistic_test_loss.npy', test_loss)
        np.save(str(dtset)+'newton_logistic_train_loss.npy', train_loss)
        np.save(str(dtset)+'newton_logistic_test_acc.npy', test_acc)
        np.save(str(dtset)+'newton_logistic_runtime.npy', runtime_list)
        np.save(str(dtset)+'newton_logistic_norm.npy', weight_norm)

        for t in range(0, T):
            t_temp = train_loss[0]*(1 + np.log(t+1)) / (t+1)
            theory_loss.append(t_temp)

        w_t = np.array(w_t)
        
        return test_loss, train_loss, test_acc, runtime_list, weight_norm