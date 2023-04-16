import numpy as np
from math import ceil

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.001, ld=0.05, batch_size=16, epochs=1000,
                replace_nan=False) -> None:
        self.lr = lr
        self.ld = ld
        self.batch_size = batch_size
        self.epochs = epochs

        self.w = None
        self.b = None
        
        self.verbosity = 1
        self.replace_nan = replace_nan

    def train(self, X, y) -> list:
        n_samples, n_features = X.shape
        y = self.reshape_y(y)

        self.w = np.zeros(shape=(n_features, 1))
        self.b = 0

        losses = []
        accuracy = []
        for epoch in range(self.epochs):
            for i_batch in range(ceil(n_samples/self.batch_size)):
                # splice data for current batch
                start_index = i_batch*self.batch_size
                end_index   = min(start_index+self.batch_size, n_samples)
                batch_n = end_index-start_index

                X_batch = X[start_index:end_index]
                y_batch = y[start_index:end_index]
                
                # calculate gradients
                preds = self.predict(X_batch)
                dw, db = self.calc_gradient(X_batch, y_batch, preds, batch_n)

                # update weights and bias using gradient descend
                self.w -= self.lr*dw
                self.b -= self.lr*db

            loss = self.calc_loss(y, self.predict(X))
            acc  = self.calc_accuracy(X, y)
            losses.append(loss)
            accuracy.append(acc)

            if self.verbosity == 1:
                print(f"epoch: {epoch}, loss: {loss}, acc: {acc}", end='\r')
            elif self.verbosity == 2:
                print(f"epoch: {epoch}, loss: {loss}, acc: {acc}")

        if self.verbosity > 0:
            print(f"training done, loss: {loss}, acc: {acc}\t\t")
        return losses, accuracy

    # loss function for binary cross entropy
    # with ridge regularization penalty
    def calc_loss(self, y_test, predictions):
        sums = y_test*np.log(predictions) + (1-y_test)*np.log(1-predictions)
        if self.replace_nan:
            sums = np.nan_to_num(sums, nan=0, posinf=0, neginf=0)
        return -np.mean(
                    sums
                ) + self.ld*np.sum(self.w**2)

    # gradient of binary cross entropy
    def calc_gradient(self, X, y, predictions, N):
        dw = (1/N)*(X.T @ (predictions-y)) + 2*self.ld*self.w
        db = (1/N)*np.sum (predictions-y)

        return dw, db
    
    def calc_accuracy(self, X_test, y_test):
        y_test = self.reshape_y(y_test)
        bin_preds = self.predict_bin(X_test)
        bin_preds = np.expand_dims(bin_preds, axis=-1)
        
        return np.sum(bin_preds==y_test)/len(bin_preds)

    def predict(self, X):
        linear_preds = X@self.w + self.b
        preds = sigmoid(linear_preds)

        return preds

    def predict_bin(self, X, threshold=0.5):
        raw_preds = self.predict(X)
        bin_preds = [1 if y>threshold else 0 for y in raw_preds]

        return bin_preds
    
    def reshape_y(self, y):
        return y.reshape((y.shape[0], 1))
    
    def set_verbosity(self, level: int=1):
        self.verbosity = level
        