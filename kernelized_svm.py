import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import seaborn as sns; sns.set()

class SMO_kernel_SVM:
    def __init__(self, kernel='linear', C=10000.0, max_iter=1000, degree=3, gamma=1):
        self.kernel = {'poly': lambda x, y: np.dot(x, y.T) ** degree,
                       'rbf': lambda x, y: np.exp(-gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1)),
                       'linear': lambda x, y: np.dot(x, y.T)}[kernel]
        self.kernel_name = kernel
        self.C = C
        self.max_iter = max_iter

        self.enter_status = 'class1'  # control enter press event
        self.X = []
        self.y = []

        self._background_figure()
        self.cid1 = self.figure.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid2 = self.figure.canvas.mpl_connect('key_press_event', self._on_press)

        plt.show()

    def labeled_coordinates(self):
        return self.X, self.y

    def alphas(self):
        return self.alphas

    def _on_press(self, event):
        if event.key == 'enter':
            if self.enter_status == 'class1':
                self.enter_status = 'class2'
                self._background_figure()

            else:
                self.X = np.array(self.X)
                self.y = np.array(self.y)
                self.enter_status = 'show_result'
                self.figure.canvas.mpl_disconnect(self.cid1)
                self.figure.canvas.mpl_disconnect(self.cid2)
                plt.cla()
                self._SMO_fit()
                self._background_figure()

        else:
            print('You pressed a wrong button')

    def _SMO_fit(self):
        self.lambdas = np.zeros_like(self.y, dtype=float)  # lambda vector whose entries are all zeros to statisfy constraints
        self.K = self.kernel(self.X, self.X) * self.y[:,np.newaxis] * self.y  # n by n kernel matrix, with elements of k(x_i, x_j)y_iy_j

        for _ in range(self.max_iter):
            for idxM in range(len(self.lambdas)):  # optimization of random pair (lam_L, lam_M)
                idxL = np.random.randint(0, len(self.lambdas))  # choose lambda_L randomly
                Q = self.K[[[idxM, idxM], [idxL, idxL]],  #    [MM ML]
                           [[idxM, idxL]              ]]  #    [LM LL] matrix
                v0 = self.lambdas[[idxM, idxL]]  # v0 = [lam_M, lam_L]^T
                k0 = 1 - np.sum(self.lambdas * self.K[[idxM, idxL]], axis=1)  # 1-sum(lam_j*k(x_M, x_j), 1-sum(lam_i*k(x_L, x_i)
                u = np.array([-self.y[idxL], self.y[idxM]])
                t_star = np.dot(k0, u) / (np.dot(np.dot(Q, u), u) + 1E-15)  # t* = (k0^Tu)/(u^TQu)
                self.lambdas[[idxM, idxL]] = v0 + u * self._restrict_to_square(t_star, v0, u)  # update new lams

            idx, = np.nonzero(self.lambdas > 1E-15)  # array of support vectors' index
            self.b = np.sum((1.0 - np.sum(self.K[idx] * self.lambdas, axis=1)) * self.y[idx]) / len(idx)  # sum_k(y_k*(1-sum_i(lam_i*K_ki)) / k

    def _SMO_decision_function(self, X):
        return np.sum(self.kernel(X, self.X) * self.y * self.lambdas, axis=1) + self.b  # classifier

    def _restrict_to_square(self, t_star, v0, u):  # restrict given t_star for lam_M(t_star), lam_L(t_star) to be [0, C]
        restricted_t_star_first_ineq = (np.clip(v0 + t_star * u, 0, self.C) - v0)[0] / u[0]
        restricted_t_star_second_ineq = (np.clip(v0 + restricted_t_star_first_ineq * u, 0, self.C) - v0)[1] / u[1]
        return restricted_t_star_second_ineq

    def _background_figure(self):
        if self.enter_status == 'class1':
            self.figure, self.ax = plt.subplots()
            self.ax.set_aspect(1)
            self.ax.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('feature 1')
            self.ax.set_ylabel('feature 2')
            self.ax.set_title('click to plot from class 1. Please press enter when finished.')
        elif self.enter_status == 'class2':
            self.ax.set_title('click to plot from class 2. please enter to start kernelized SVM with SMO.')
        elif self.enter_status == 'show_result':
            self.ax.set_aspect(1)
            self.ax.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('feature 1')
            self.ax.set_ylabel('feature 2')
            self.ax.set_title(f'{self.kernel_name} SVM with SMO (C={self.C})')

            # kernel SVM with SMO plot
            xx, yy = np.meshgrid(np.linspace(*[-5, 5], num=700), np.linspace(*[-5, 5], num=700))
            z_model = self._SMO_decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=50, cmap=ListedColormap(['r', 'b']))
            plt.contour(xx, yy, z_model, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])  # support vector line
            plt.contourf(xx, yy, np.sign(z_model), alpha=0.3, levels=2, cmap=ListedColormap(["darkred", "darkblue"]), zorder=1)

    def _on_click(self, event):
        if self.enter_status == 'class1':
            self._draw_click(event)
            self.X.append([event.xdata, event.ydata])
            self.y.append(1)
            print(f'Coordinate: [{event.xdata}, {event.ydata}], Label: {1}')
        else:
            self._draw_click(event)
            self.X.append(([event.xdata, event.ydata]))
            self.y.append(-1)
            print(f'Coordinate: [{event.xdata}, {event.ydata}], Label: {-1}')

    def _draw_click(self, event):
        if self.enter_status == 'class1':
            self.ax.scatter(event.xdata, event.ydata, marker='.', c='r')
        else:
            self.ax.scatter(event.xdata, event.ydata, marker='x', c='b')