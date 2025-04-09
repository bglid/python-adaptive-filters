import numpy as np


# Eigendecomposition and Power Iteration
class EigenDecomposition:
    def __init__(self, k_iterations):
        self.iterations = k_iterations

    def power_iteration(self, X):
        # initializing y as a random vector of a matching shape to X[0]
        X_shape = X.shape[0]
        y = np.random.rand(X_shape)
        # iterating and updating for k_iterations:
        for k in range(self.iterations):
            product = np.dot(X, y)
            # updating y with normalized product
            y = product / np.linalg.norm(product)
        return y

    def get_eigenval(self, eigenvector, X):
        # returning eigenval
        return np.dot(eigenvector.T, np.dot(X, eigenvector))

    # function that gets a subsequent EV given a deflated matrix
    def get_eigenvectors(self, eigenvector, eigenval, X):
        deflated_matrix_n = X - eigenval * np.outer(eigenvector, eigenvector.T)
        # retuning the deflated matrix at n for next calculation
        return self.power_iteration(deflated_matrix_n), deflated_matrix_n

    # process that gets X num of EV using above functions
    def eigendecomposition(self, covariance_matrix, n_eigenvectors):
        # getting first EV using just power_iteration
        ev_1 = self.power_iteration(covariance_matrix)
        # we will be stacking into a matrix, so reshaping ev_1
        w_matrix = ev_1.reshape(-1, 1)
        # creating our diagonal matrix:
        lambda_matrix = np.zeros(shape=((n_eigenvectors, n_eigenvectors)))
        # we will be iterating using the previous EV_n, setting EV1 to start
        ev_n = ev_1
        # copying cov_matrix for getting next EV
        deflated_matrix = covariance_matrix.copy()
        eigenvals = []
        for n in range(n_eigenvectors - 1):  # -1 since we already have EV# 1
            lambda_n = self.get_eigenval(ev_n, covariance_matrix)
            lambda_matrix[n][n] = float(lambda_n)

            print(f"Eigenval #{n + 1}: {lambda_n}")
            if type(lambda_n) is np.ndarray:
                eigenvals.append(lambda_n[0][0])
            else:
                eigenvals.append(lambda_n)

            # getting next eigenvector and updated deflated matrix
            ev_n, deflated_matrix = self.get_eigenvectors(
                ev_n, lambda_n, deflated_matrix
            )
            ev_n = ev_n.reshape(-1, 1)
            w_matrix = np.hstack((w_matrix, ev_n))

        # logging and adding final Eigenval
        final_lambda = self.get_eigenval(ev_n, covariance_matrix)
        eigenvals.append(final_lambda[0][0])
        lambda_matrix[n_eigenvectors - 1][n_eigenvectors - 1] = final_lambda

        print(
            f"Eigenval #{n_eigenvectors}: {self.get_eigenval(ev_n, covariance_matrix)}"
        )
        # adding eigenvals for graphing
        return w_matrix, lambda_matrix, eigenvals
