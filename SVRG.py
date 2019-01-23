class SVRG(Opti_algo):
    def train(self):
        # lambda, T
        T = 5000
        M = 200
        epsilon = 0.05
        S = train_labels.shape[1]
        test_loss = []
        train_loss = []
        theory_loss = []
        test_acc = []
        weight_norm = []
        runtime_list = []
        w_tt = np.random.randn(1, train_imgs.shape[2]*train_imgs.shape[3])  ## current w. 
        w_mean = np.zeros((1, train_imgs.shape[2]*train_imgs.shape[3]))
        w_tt = np.array(w_tt)
        w_t=[]    ## list of all weights.
        w_t.append(w_tt)

        X = np.reshape(train_imgs[0,:,:,:], (-1, train_imgs.shape[2]*train_imgs.shape[3]))/255

        start_sgd = timeit.default_timer()
        # Start Iteration Loop
        for t in range(0, T):
            if t % 100 == 0:
                print("running", t)

            w_bar = w_tt
            mu = self.obj.cal_full_grad(X, w_bar)

            w_prev = w_bar

            for l in range(1,M):
                step_t = 0.01 ## ?? 
                # Select random it
                i_t = np.random.randint(S, size=1)
                # Pickout x_it
                x_it = np.reshape(train_imgs[0,i_t,:,:], (1, train_imgs.shape[2]*train_imgs.shape[3]))/255 # X[i_t,:]
                y_it = y_from_label_train(i_t)

                sto_grad1 = self.obj.cal_gradient(x_it, w_prev, y_it)
                sto_grad2 = self.obj.cal_gradient(x_it, w_bar, y_it)

                w_curr = w_prev - step_t*(sto_grad1 - sto_grad2 + mu)
                w_prev = w_curr
                # Calculate step size

            w_tt = w_curr

            w_mean = (w_mean * t + w_tt)/(t+1)

            # Append test error
            if t % 100 == 0:
                stop_sgd = timeit.default_timer()
                runtime = stop_sgd - start_sgd
                test_loss.append(self.obj.test_error(w_mean))
                train_loss.append(self.obj.train_error(w_mean))
                test_acc.append(self.obj.cal_test_acc(w_mean))
                weight_norm.append(w_mean)
                runtime_list.append(runtime)
                print("test_acc", test_acc[-1])

        np.save('cfar_sgd_logi_svrg_test_loss.npy', test_loss)
        np.save('cfar_sgd_logi_svrg_train_loss.npy', train_loss)
        np.save('cfar_sgd_logi_svrg_test_acc.npy', test_acc)
        np.save('cfar_sgd_logi_svrg_runtime.npy', runtime_list)
        np.save('cfar_sgd_logi_svrg_norm.npy', weight_norm)

        for t in range(0, T//100):
            t_temp = train_loss[0] / ((t+1)*100)
            theory_loss.append(t_temp)

        w_t = np.array(w_t)
        theory_loss += np.min(test_loss)
        
        return test_loss, train_loss, test_acc, runtime_list, weight_norm
    