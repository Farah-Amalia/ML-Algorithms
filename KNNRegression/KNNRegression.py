# Importing libraries
import numpy as np

# Creating class
class KNNRegression() :
	def __init__(self, K) :
		self.K = K
	
  # Storing training dataset	
	def fit(self, X_train, Y_train) :	
		self.X_train = X_train		
		self.Y_train = Y_train		
		# Define number of training samples and number of features		
		self.m, self.n = X_train.shape	
	
  # Function for prediction		
	def predict(self, X_test) :		
		self.X_test = X_test		
		# Define number of testing samples and number of features		
		self.m_test, self.n = X_test.shape		
		# Initialize Y_predict		
		Y_predict = np.zeros(self.m_test)
		# Loop through testing dataset		
		for i in range(self.m_test) :			
			x = self.X_test[i]			
			# Find the K nearest neighbors from current test example			
			neighbors = np.zeros(self.K)			
			neighbors = self.find_neighbors(x)			
			# Calculate the mean of K nearest neighbors			
			Y_predict[i] = np.mean(neighbors)			
		return Y_predict	
	
  # Function to find the K nearest neighbors to current test example			
	def find_neighbors(self, x) :		
		# Calculatr distance
		euclidean_distances = np.zeros(self.m)	
		# Looping through each training data	
		for i in range(self.m) :			
			d = self.euclidean(x, self.X_train[i])			
			euclidean_distances[i] = d		
		# Sorting array and preserving index	
		inds = euclidean_distances.argsort()		
		Y_train_sorted = self.Y_train[inds]		
		return Y_train_sorted[:self.K]	
	
  # Function to calculate euclidean distance			
	def euclidean(self, x, x_train):		
		return np.sqrt(np.sum(np.square(x - x_train)))