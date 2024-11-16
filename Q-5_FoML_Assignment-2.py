import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data paths are given as per file locations in PC
#Training Features
gisette_train_data=open('D:\MTECH\Sem-1\Foundations of ML\GISETTE_2\gisette_train.data',"r")
training_features=[]
for row in gisette_train_data.readlines():
    training_features.append((row.strip()).split(" "))
gisette_train_data.close()

#Training Labels
gisette_train_labels= open("D:\MTECH\Sem-1\Foundations of ML\GISETTE_2\gisette_train.labels")
training_labels=[]
for row in gisette_train_labels.readlines():
    training_labels.append((row.strip()).split(" "))
gisette_train_labels.close()

#Testing Features
gisette_valid_data=open('D:\MTECH\Sem-1\Foundations of ML\GISETTE_2\gisette_valid.data',"r")
testing_features=[]
for row in gisette_valid_data.readlines():
    testing_features.append((row.strip()).split(" "))
gisette_valid_data.close()

#Testing Labels
gisette_valid_labels= open("D:\MTECH\Sem-1\Foundations of ML\GISETTE_2\gisette_valid.labels")
testing_labels=[]
for row in gisette_valid_labels.readlines():
    testing_labels.append((row.strip()).split(" "))
gisette_valid_labels.close()

X_train = pd.DataFrame(training_features)
Y_train = pd.DataFrame(training_labels)

X_test = pd.DataFrame(testing_features)
Y_test=  pd.DataFrame(testing_labels)

#5.a Linear Kernel Implementation
print("**************************************************************************")
print("5.a Linear Kernel Implementation")
linear_kernel = svm.SVC(kernel='linear',C=1.0,gamma='auto')
linear_kernel.fit(X_train,Y_train.values.ravel())
linear_predict1 = linear_kernel.predict(X_train)
linear_train_accuracy = accuracy_score(Y_train,linear_predict1)
linear_predict2 = linear_kernel.predict(X_test)
linear_test_accuracy = accuracy_score(Y_test,linear_predict2)
print("Training error is: " + str(round((1-linear_train_accuracy),5))," Testing error is: " + str(round((1-linear_test_accuracy),5)))
print("Total Support vectors are :",len(linear_kernel.support_vectors_))

#5.b RBF Kernel Implementation
print("**************************************************************************")
print("5.b RBF Kernel Implementation")
rbf_kernel = svm.SVC(kernel = 'rbf',C=1.0,gamma=0.001)
rbf_kernel.fit(X_train,Y_train.values.ravel())
rbf_predict1 = rbf_kernel.predict(X_train)
rbf_train_accuracy = accuracy_score(Y_train,rbf_predict1)
rbf_predict2 = rbf_kernel.predict(X_test)
rbf_test_accuracy = accuracy_score(Y_test,rbf_predict2)
print("Training error is: " + str(round((1-rbf_train_accuracy),5))," Testing error is: " + str(round((1-rbf_test_accuracy),5)))
print("Total Support vectors are :",len(rbf_kernel.support_vectors_))

#5.b Polynomial Kernel Implementation
print("**************************************************************************")
print("5.b Polynomial Kernel Implementation")
polynomial_kernel = svm.SVC(kernel = 'poly',C=1.0,degree=2,gamma='auto')
polynomial_kernel.fit(X_train,Y_train.values.ravel())
poly_predict1 = polynomial_kernel.predict(X_train)
poly_train_accuracy = accuracy_score(Y_train,poly_predict1)
poly_predict2 = polynomial_kernel.predict(X_test)
poly_test_accuracy = accuracy_score(Y_test,poly_predict2)
print("Training error is: " + str(round((1-poly_train_accuracy),5))," Testing error is: " + str(round((1-poly_test_accuracy),5)))
print("Total Support vectors are :",len(polynomial_kernel.support_vectors_))

print("**************************************************************************")
