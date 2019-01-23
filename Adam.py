class Adam(Opti_algo):
    def train(self):
        T = 5000
        alpha = 0.1
        beta_1 = 0.9
        beta_2 = 0.99  #initialize the values of the parameters
        epsilon = 0.1

        epsilon = 0.05
        S = train_labels.shape[1]
        test_loss = []
        train_loss = []
        theory_loss = []
        test_acc = []
        weight_norm = []
        runtime_list = []
        w_tt = np.random.randn(1, train_imgs.shape[2]*train_imgs.shape[3])
        w_mean = np.zeros((1, train_imgs.shape[2]*train_imgs.shape[3]))
        w_tt = np.array(w_tt)
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
            y_prev = w_tt
            alpha = 0.05 / (2 * (t+1))
            g_it = self.obj.cal_gradient(x_it, w_tt, y_it)

            m_t = 0
            v_t = 0
            ti = 0
            w_tt_prev = 0
            while np.linalg.norm(g_it) >= epsilon:  #till it gets converged
                ti += 1
                g_it = self.obj.cal_gradient(x_it, w_tt, y_it)  
                m_t = beta_1*m_t + (1-beta_1)*g_it  
                v_t = beta_2*v_t + (1-beta_2)*(g_it*g_it)  
                m_cap = m_t/(1-(beta_1**ti))  
                v_cap = v_t/(1-(beta_2**ti))  
                w_tt_prev = w_tt
                w_tt = w_tt - (alpha*m_cap)/(np.sqrt(v_cap)+step_t)	

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

        

        np.save('cfar_sgd_logi_adam_test_loss.npy', test_loss)
        np.save('cfar_sgd_logi_adam_train_loss.npy', train_loss)
        np.save('cfar_sgd_logi_adam_test_acc.npy', test_acc)
        np.save('cfar_sgd_logi_adam_runtime.npy', runtime_list)
        np.save('cfar_sgd_logi_adam_norm.npy', weight_norm)

        for t in range(0, T//100):
            t_temp = train_loss[0]/ (100*(t+1))
            theory_loss.append(t_temp)

        theory_loss += np.min(test_loss)
        
        return test_loss, train_loss, test_acc, runtime_list, weight_norm