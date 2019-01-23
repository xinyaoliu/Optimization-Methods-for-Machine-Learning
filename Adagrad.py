class Adagrad(Opti_algo):
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
        G_matrix = np.zeros((1, w_tt.shape[1]))
        smooth_epsilon = 1e-3

        test_acc = []
        start_sgd = timeit.default_timer()
        runtime_list = []
        weight_norm = []
        # Start Iteration Loop
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
            # Update the G matrix
            G_matrix += g_it ** 2
            # Modift gradient
            g_it = g_it / (np.sqrt(smooth_epsilon + G_matrix))

            # Do w_t update
            w_tt = w_tt - step_t * g_it # size 1 by N
            # lambda
            # Do w_t update
            if(np.linalg.norm(w_tt)>lamba):
                w_tt = lamba * w_tt/np.linalg.norm(w_tt)

            w_t.append(w_tt)
            w_mean = (w_mean * t + w_tt)/(t+1)
    

            if t%100 == 0:

                stop_sgd = timeit.default_timer()
                runtime  = stop_sgd - start_sgd
                print("test accuracy:", self.obj.cal_test_acc(w_mean))
                test_loss.append(self.obj.test_error(w_mean))
                train_loss.append(self.obj.train_error(w_mean))
                test_acc.append(self.obj.cal_test_acc(w_mean))
                runtime_list.append(runtime)
                weight_norm.append(w_mean)


        np.save(str(dtset)+'Adagrad_logistic_test_loss.npy', test_loss)
        np.save(str(dtset)+'Adagrad_logistic_train_loss.npy', train_loss)
        np.save(str(dtset)+'Adagrad_logistic_test_acc.npy', test_acc)
        np.save(str(dtset)+'Adagrad_logistic_runtime.npy', runtime_list)
        np.save(str(dtset)+'Adagrad_logistic_norm.npy', weight_norm)


        for t in range(0, T):
            t_temp = train_loss[0]*(1 + np.log(t+1)) / (t+1)
            theory_loss.append(t_temp)

        w_t = np.array(w_t)
        theory_loss += np.min(test_loss)
        
        return test_loss, train_loss, test_acc, runtime, weight_norm