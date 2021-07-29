# Import required module
import numpy as np
 
# Creating class
class LinearRegression:
    def __init__(self, cost_function="l2", learning_rate=0.01, epochs=1000000) :
      self.learning_rate = learning_rate
      self.epochs = epochs
      self.cost_function = cost_function
    # Model fitting
    def fit(self, X, Y):
        # Define number of training samples and number of features        
        self.m, self.n = X.shape          
        # Weight initialization          
        self.W = np.zeros(self.n)          
        self.b = 0          
        self.X = X          
        self.Y = Y 
        # Implementing Gradient Descent
        # for Mean Square Error
        if self.cost_function == "l2":
            for i in range(self.epochs):
                Y_pred = self.predict(self.X) 
                # Calculating derivatives        
                dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m       
                db = - 2 * np.sum(self.Y - Y_pred) / self.m           
                # Updating parameters      
                self.W = self.W - self.learning_rate * dW      
                self.b = self.b - self.learning_rate * db 
        elif self.cost_function == "l1":
            for i in range(self.epochs):
                Y_pred = self.predict(self.X) 
                # Calculating derivatives        
                dW = - (1/self.m) * np.sum(((self.X.T).dot(self.Y - Y_pred)) / abs(self.Y - Y_pred))      
                db = - (1/self.m) * np.sum((self.Y - Y_pred) / abs(self.Y - Y_pred))           
                # Updating parameters      
                self.W = self.W - self.learning_rate * dW      
                self.b = self.b - self.learning_rate * db 
        self.coef_ = self.W
        self.intercept_ = self.b
        return self

    # Prediction              
    def predict(self, X):      
        Y_pred = X.dot(self.W) + self.b
        return Y_pred

    