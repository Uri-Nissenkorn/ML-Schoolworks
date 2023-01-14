#################################
# Your name: Uri Nissenkorn
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        sample = np.array([[x,self.P_yIx(x)] for x in np.random.random(m)])
        return sample[sample[:,0].argsort()]

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        errors = np.zeros([(m_last-m_first)//step, 2])
        j = 0
        for m in range(m_first,m_last,step):
            sum_ep = 0
            sum_et = 0
            for i in range(T):
                sample = self.sample_from_D(m)
                inters , curr_ep = intervals.find_best_interval(sample[:,0],sample[:,1],k)
                sum_ep += curr_ep / m

                sum_et += self.intervals_true_error(inters)
            
            avg_ep = sum_ep / T
            avg_et = sum_et / T

            errors[j,0] = avg_ep
            errors[j,1] = avg_et

            j+=1

        x = np.arange(m_first,m_last,step)
        plt.plot(x, errors[:,0], label='Empirical Error')
        plt.plot(x, errors[:,1], label='True Error')
        plt.legend()
        plt.xlabel("n")
        plt.ylabel("Error Rate")
        plt.show()
        return errors


    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """

        errors = np.zeros([(k_last-k_first)//step,2])
        sample = self.sample_from_D(m)

        i = 0
        for k in range(k_first,k_last,step):
            inters , curr_ep = intervals.find_best_interval(sample[:,0],sample[:,1],k)
            errors[i,0] = curr_ep /m
            errors[i,1] =  self.intervals_true_error(inters) 
            i+=1

        x = np.arange(k_first,k_last,step)
        plt.plot(x, errors[:,0], label='Empirical Error')
        plt.plot(x, errors[:,1], label='True Error')
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("Error Rate")
        plt.show()
        return np.argmin(errors[:,0]+1)

            

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        errors = np.zeros([(k_last-k_first)//step, 4])
        sample = self.sample_from_D(m)
        delta = 0.1

        i = 0
        for k in range(k_first,k_last,step):
            inters , curr_ep = intervals.find_best_interval(sample[:,0],sample[:,1],k)
            vcdim = 2*k
                
            errors[i,0] = curr_ep / m
            errors[i,1] = self.intervals_true_error(inters)
            errors[i,2] = np.sqrt((vcdim-np.log(1/delta))/m)
            errors[i,3] = errors[i,0]+errors[i,2]
            
            i+=1

        x = np.arange(k_first,k_last,step)
        plt.plot(x, errors[:,0], label='Empirical Error')
        plt.plot(x, errors[:,1], label='True Error')
        plt.plot(x, errors[:,2], label='Penalty')
        plt.plot(x, errors[:,3], label='Empirical Error + Penalty')
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("Error Rate")
        plt.show()
        return np.argmin(errors[:,3]+1)

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)
        np.random.shuffle(sample)

        test = sample[:m//5] # take 0.2 
        train = sorted(sample[m//5:], key=lambda x: x[0])

        x_train = [i[0] for i in train]
        y_train = [i[1] for i in train]
        x_test = [i[0] for i in test]
        y_test = [i[1] for i in test]

        errors = []
        for k in range(1, 11):
            # train 
            inters , curr_ep = intervals.find_best_interval(x_train, y_train, k)

            # test
            error = 0
            for i in range(len(x_test)):
                if self.predict_x(x_test[i], inters) != y_test[i]:
                    error+=1
            errors.append(error/len(y_test))
        return np.argmin(errors)+1

    #################################
    # Place for additional methods
    def P_yIx(self,x):
        if 0<=x<=0.2 or 0.4<=x<=0.6 or 0.8<=x<=1:
            return np.random.choice([0,1],size=1, p=[0.2,0.8])[0]
        else:
            return np.random.choice([0,1],size=1, p=[0.9,0.1])[0]


    def intervals_true_error(self, intervals):
        error = 0

        for i in intervals:
            error+=self.interval_true_error(i , 1)

        reverse_intervals = self.get_reverse_intervals(intervals)
        for i in reverse_intervals:
            error+=self.interval_true_error(i , 0)

        return error
        

    def interval_true_error(self, interval, prediction):
        if prediction == 1:
            with_1 = max([self.intrevals_cross(interval,x) for x in [(0,0.2),(0.4,0.6),(0.8,1)]]) * (1-0.8)
            with_0 = max([self.intrevals_cross(interval,x) for x in [(0.2,0.4),(0.6,0.8)]]) * (1-0.1)
            return with_1 + with_0
        else:
            with_1 = max([self.intrevals_cross(interval,x) for x in [(0,0.2),(0.4,0.6),(0.8,1)]]) * (0.8)
            with_0 = max([self.intrevals_cross(interval,x) for x in [(0.2,0.4),(0.6,0.8)]]) * (0.1)
            return with_1 + with_0


    def intrevals_cross(self, i1, i2):
        if i1[1]<i2[0] or i2[1]<i1[0]:
            return 0
        return min(i1[1],i2[1]) - max(i1[0],i2[0])

    
    def get_reverse_intervals(self, intervals):
        start = 0

        temp1 = 0
        temp2 = 0

        r_inrtervals = []

        if intervals[0][0] == 0:
            if intervals[0][1] == 1:
                return []
            temp1 = intervals[0][1]
            start = 1

        for i in range(start,len(intervals)):
            temp2 = intervals[i][0]
            r_inrtervals.append((temp1,temp2))

            temp1 = intervals[i][1]
        
        if temp1 == 1:
            return r_inrtervals

        temp2 = 1
        r_inrtervals.append((temp1,temp2))
        return r_inrtervals

    def predict_x(self, x, intervals):
        for interval in intervals:
            if interval[0] <= x <= interval[1]:
                return 1
            elif x >= interval[1]:
                return 0
        return 0

    #################################


if __name__ == '__main__':
    ass = Assignment2()

    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

