

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/Dell/Documents/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Data.csv')

#Splitting a dataset into two different dataframes
X = dataset.iloc[:, :-1].values
df_X = pd.DataFrame(X)
Y = dataset.iloc[:, 3].values
df_Y = pd.DataFrame(Y)

# # Taking care of missing data
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:, 1:3])


# # Encoding categorical data.

# # here LabelEncoder class is used for encoding the data which is present in the string form to numbers.
# # where as OnHotEncoder class is used to divide country column into 3 different columns, so that labelencoder do not
# # put the order of country column data by itself.

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.compose import ColumnTransformer

# # For conntry column
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0]) 
# X = onehotencoder.fit_transform(X).toarray()

# ct = ColumnTransformer(
#     [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
#     remainder='passthrough'                         # Leave the rest of the columns untouched
# )

# X = np.array(ct.fit_transform(X), dtype=np.float)

# # For purchased coulumn
# labelencoder_Y = LabelEncoder()
# Y = labelencoder_Y.fit_transform(Y)

# # Splitting the dataset into training set and a test set

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# #Feature Scaling

# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)










