class SGD(Opti_algo):
    
    def train(self):
        # lambda, T
        T = 1000
        lamba = 1
        S = train_labels.shape[1]
        test_loss = []
        train_loss = []
        test_acc = []
        theory_loss = []
        weight_norm = []
        w_tt = np.random.randn(1, train_imgs.shape[2]*train_imgs.shape[3])
        w_mean = np.zeros((1, train_imgs.shape[2]*train_imgs.shape[3]))
        w_tt = np.array(w_tt)
        w_t=[]
        w_t.append(w_tt)
        runtime_list = []
        # Start Iteration Loop
        start_sgd = timeit.default_timer()
        for t in range(0, T):
            if t % 100 == 0:
                print("running", t)

            # Select random it
            i_t = np.random.randint(S, size=1)
            # Calculate step size
            step_t = 0.01 # modify
            # Locate label
            y_it = y_from_label_train(i_t)
            # Pickout x_it
            x_it = np.reshape(train_imgs[0,i_t,:,:], (1, train_imgs.shape[2]*train_imgs.shape[3]))/255
            # Do inner production
            g_it = self.obj.cal_gradient(x_it, w_tt, y_it)

            # Do w_t update
            w_tt = w_tt - step_t * g_it # size 1 by N
            # lambda
            # Do w_t update
            '''
            if(np.linalg.norm(w_tt)>lamba):
                w_tt = lamba * w_tt/np.linalg.norm(w_tt)
            '''
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
                

        stop_sgd = timeit.default_timer()
        runtime = stop_sgd - start_sgd

        np.save(str(dtset)+'sgd_l1_test_loss.npy', test_loss)
        np.save(str(dtset)+'sgd_l1_train_loss.npy', train_loss)
        np.save(str(dtset)+'sgd_l1_test_acc.npy', test_acc)
        np.save(str(dtset)+'sgd_l1_runtime.npy', runtime_list)
        np.save(str(dtset)+'sgd_l1_norm.npy', weight_norm)

        for t in range(0, T//100):
            t_temp = train_loss[0] / (100*(t+1))
            theory_loss.append(t_temp)

        w_t = np.array(w_t)
        theory_loss += np.min(test_loss)
        
        return test_loss, train_loss, test_acc, runtime_list, weight_norm