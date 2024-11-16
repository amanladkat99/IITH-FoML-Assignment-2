import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data paths are given as per file locations in PC
#Initial Training Data
training_data = pd.read_csv("D:\MTECH\Sem-1\Foundations of ML\Datasets\digitTrain.txt",sep='  ',engine='python',header=None)
training_data.columns = ["training_labels", "features1_train", "features2_train"]

#Filtered Training Data
filtered_train_data = training_data[training_data["training_labels"].isin([1,5])]

#Initial Testing Data
testing_data = pd.read_csv("D:\MTECH\Sem-1\Foundations of ML\Datasets\digitTest.txt",sep='  ',engine='python',header=None)
testing_data.columns = ["testing_labels", "features1_test", "features2_test"]

#Filtered Testing Data
filtered_testing_data = testing_data[testing_data["testing_labels"].isin([1,5])]

features_train = filtered_train_data[['features1_train', 'features2_train']]
training_label = filtered_train_data['training_labels']
X_train = features_train.values
Y_train = training_label.values

features_test = filtered_testing_data[['features1_test', 'features2_test']]
testing_label = filtered_testing_data['testing_labels']
X_test = features_test.values
Y_test = testing_label.values

#4.a Linear Kernel Implementation
print("**************************************************************************")
print("4.a Linear Kernel Implementation")
linear_kernel = svm.SVC(kernel='linear',C=1.0,gamma='auto')
linear_kernel.fit(X_train,Y_train)
linear_predict = linear_kernel.predict(X_test)
linear_accuracy = accuracy_score(Y_test,linear_predict)
print("Test Accuracy of Soft Margin Linear SVM is : " + str(round((linear_accuracy),5)))
print("Total number of Support vectors are :",len(linear_kernel.support_vectors_))

#4.b
print("**************************************************************************")
print("4.b")
no_of_samples = [50,100,200,800]
for num in no_of_samples:
    portion_of_X_train = X_train[:num]
    portion_of_Y_train = Y_train[:num]
    portion_of_X_test = X_test[:num]
    portion_of_Y_test = Y_test[:num]

    linear_kernel = svm.SVC(kernel='linear',C=1.0,gamma='auto')
    linear_kernel.fit(portion_of_X_train,portion_of_Y_train)
    linear_predict = linear_kernel.predict(portion_of_X_test)
    linear_accuracy = accuracy_score(portion_of_Y_test,linear_predict)
    print("Samples:",num, "Test Accuracy is: " + str(round((linear_accuracy),5)), "Total Support vectors are :",len(linear_kernel.support_vectors_))

#4.c Polynomial Kernel Implementation
print("**************************************************************************")
print("4.c Polynomial Kernel Implementation")
Q = [2, 5]
for num in Q:
    for exp in range(0, 5):
        c_value = 1 * pow(10, -exp)
        polynomial_kernel = svm.SVC(kernel = 'poly',C=c_value,degree=num,gamma='auto')
        polynomial_kernel.fit(X_train,Y_train)
        poly_predict1 = polynomial_kernel.predict(X_train)
        poly_train_accuracy = accuracy_score(Y_train,poly_predict1)
        poly_predict2 = polynomial_kernel.predict(X_test)
        poly_test_accuracy = accuracy_score(Y_test,poly_predict2)
        print("Q:",num,"Cval:",c_value," Training error is: " + str(round((1-poly_train_accuracy),5))," Testing error is: " + str(round((1-poly_test_accuracy),5))," Total Support vectors are :",len(polynomial_kernel.support_vectors_))

#4.d  RBF Kernel Implementation
print("**************************************************************************")
print("4.d RBF Kernel Implementation")        
values_of_c = [0.01,1,100,10**4,10**6]
for c_value in values_of_c:
    rbf_kernel = svm.SVC(kernel = 'rbf',C=c_value, gamma='auto')
    rbf_kernel.fit(X_train,Y_train)
    rbf_predict1 = rbf_kernel.predict(X_train)
    rbf_train_accuracy = accuracy_score(Y_train,rbf_predict1)
    rbf_predict2 = rbf_kernel.predict(X_test)
    rbf_test_accuracy = accuracy_score(Y_test,rbf_predict2)
    print("Cval:",c_value," Training error is: " + str(round((1-rbf_train_accuracy),5))," Testing error is: " + str(round((1-rbf_test_accuracy),5)))

print("**************************************************************************")
