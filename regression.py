import numpy as np

class Linear_Regression:
  def __init__(self, X, Y):
    self.x = X
    self.y = Y

  def m(self, X): 
    return sum(X)/len(X)

  def d(self, X):
    return self.m(X**2) - (self.m(X))**2

  def correlation(self, X, Y):
    return sum((X - self.m(X)) * (Y - self.m(Y))) / (len(X) * (self.d(X) * self.d(Y)) ** 0.5)

  def fit(self):
    self.a = (self.m(self.y)*self.m(self.x)-(self.m(self.y*self.x)))/(self.m(self.x)**2-self.m(self.x**2))
    self.b = self.m(self.y)-self.a*self.m(self.x)

  def predict(self, X):
    return self.a*X + self.b