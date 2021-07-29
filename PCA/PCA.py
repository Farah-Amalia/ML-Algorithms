# Import required module
import numpy as np

# Creating class 
class PCA:
    def __init__(self,n_components):
      self.n_components = n_components
    # Fitting
    def fit(self,X): 
      # Substracting the mean
      self.X_meaned = X - np.mean(X , axis = 0)
      
      # Calculating covariance matrix
      cov_mat = np.cov(self.X_meaned , rowvar = False)
      
      # Computing eigen values and eigen vectors
      eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
      
      # Sorting eigen values and eigen values
      sorted_index = np.argsort(eigen_values)[::-1]
      sorted_eigenvalue = eigen_values[sorted_index]
      sorted_eigenvectors = eigen_vectors[:,sorted_index]
      
      # Subsetting
      self.eigenvector_subset = sorted_eigenvectors[:,0:self.n_components]

      self.components_ = self.eigenvector_subset.transpose()
      self.explained_variance_ = sorted_eigenvalue[:-1]

    # Transform
    def transform(self,X): 
      X_reduced = np.dot(self.eigenvector_subset.transpose(), self.X_meaned.transpose() ).transpose()
      return X_reduced