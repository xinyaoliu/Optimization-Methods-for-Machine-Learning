class BFGS(Opti_algo):
    def train(self):
        T = 5000
        lamba = 1
        S = train_labels.shape[1]
        test_loss = []
        train_loss = []
        theory_loss = []
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
        # Define values for BFGS
        i_t = np.random.randint(S, size=1)
        y_it = y_from_label_train(i_t)
        x_it = np.reshape(train_imgs[0,i_t,:,:], (1, train_imgs.shape[2]*train_imgs.shape[3]))/255
        H_tt = np.identity(train_imgs.shape[2]*train_imgs.shape[3])
        BFGS_g_former = self.obj.cal_gradient(x_it, w_tt, y_it)
        #print(BFGS_g_former)
        BFGS_w_former= w_tt
        identity_matrix = np.identity(train_imgs.shape[2]*train_imgs.shape[3])
        lambda_ = 1
        start_sgd = timeit.default_timer()
        runtime_list = []
        test_acc = []
        weight_norm=[]


        # Start Iteration Loop
        for t in range(0, T):
            if t % 100 == 0:
                print("running", t)

            # Select random it
            i_t = np.random.randint(S, size=1)
            # Calculate step size
            step_t = 1 # modify
            # Locate label
            y_it = y_from_label_train(i_t)
            # Pickout x_it
            x_it = np.reshape(train_imgs[0,i_t,:,:], (1, train_imgs.shape[2]*train_imgs.shape[3]))/255
            w_tt = BFGS_w_former - step_t*(np.matmul(H_tt, self.obj.cal_gradient(x_it, BFGS_w_former, y_it).T).T)
            BFGS_g = self.obj.cal_gradient(x_it, w_tt, y_it)
            BFGS_s = (w_tt - BFGS_w_former)
            BFGS_y = BFGS_g - self.obj.cal_gradient(x_it, BFGS_w_former, y_it)+lambda_*BFGS_s
            if t == 0:
                H_tt = np.matmul(BFGS_s,BFGS_y.T)/np.matmul(BFGS_y,BFGS_y.T)*identity_matrix
            # update H_tt
            '''
            scale_parameter_s = (np.sum(BFGS_s*BFGS_s)+1e-7)
            BFGS_s = BFGS_s/scale_parameter_s
            scale_parameter_y = (np.sum(BFGS_y*BFGS_y)+1e-7)
            BFGS_y = BFGS_y/scale_parameter_y
            '''
            rescale_value = np.matmul(BFGS_y,BFGS_s.T)+1e-7
            H_tt = np.matmul(np.matmul((identity_matrix - np.matmul(BFGS_s.T,BFGS_y)/rescale_value),H_tt),(identity_matrix-np.matmul(BFGS_y.T,BFGS_s)/rescale_value))+np.matmul(BFGS_s.T,BFGS_s)/rescale_value
           
            BFGS_g_former = BFGS_g
            BFGS_w_former = w_tt
            
            w_t.append(w_tt)
            w_mean = (w_mean * t + w_tt)/(t+1)

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


        np.save(str(dtset)+'BFGS_logistic_test_loss.npy', test_loss)
        np.save(str(dtset)+'BFGS_logistic_train_loss.npy', train_loss)
        np.save(str(dtset)+'BFGS_logistic_test_acc.npy', test_acc)
        np.save(str(dtset)+'BFGS_logistic_runtime.npy', runtime_list)
        np.save(str(dtset)+'BFGS_logistic_norm.npy', weight_norm)

        for t in range(0, T):
            t_temp = train_loss[0]*(1 + np.log(t+1)) / (t+1)
            theory_loss.append(t_temp)

        w_t = np.array(w_t)
        theory_loss += np.min(test_loss)
        
        return test_loss, train_loss, test_acc, runtime_list, weight_norm