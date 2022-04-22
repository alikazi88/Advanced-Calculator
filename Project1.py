#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt  # To visualize


# In[6]:


# Printing the log base 5 of 14
def find_log_with_base():
    try:
        print("Calculating log with base and number of your choice")
        add = input("Have this Logarithm(1+a) to compute? Y/N: ")
        add = str(add)
        num = input("Enter the number: ")
        num = float(num)
        base = input("Enter the base.If none, put 1 : ")
        base = float(base)

        if add == 'Y':
            print ("Logarithm(1+a) : ", end="")
            print (math.log1p(num))
        if base ==2:
            print ("Logarithm base 2 : ", end="")
            print (math.log2(num))
        if base == 10:
            print ("Logarithm base 10 : ", end="")
            print (math.log10(num))
        if base == 1:
            print ("Logarithm is : ", end="")
            print (math.log(num))
    except ValueError:
        print("")
        print("Number is negative")


# In[59]:


find_log_with_base()


# In[67]:


# A basic code for matrix input from user
def find_matrix():
    R = int(input("Enter the number of rows:"))
    C = int(input("Enter the number of columns:"))

    # Initialize matrix
    matrix = []
    print("Enter the entries rowwise:")

    # For user input
    for i in range(R):		 # A for loop for row entries
        a =[]
        for j in range(C):	 # A for loop for column entries
            a.append(int(input()))
        matrix.append(a)

    # For printing the matrix
    for i in range(R):
        for j in range(C):
            print(matrix[i][j], end = " ")
        print()

    n_array = np.array(matrix)
    det = np.linalg.det(n_array)
    print("Determinant of given matrix:",det)


# In[69]:


find_matrix()


# In[76]:


def find_eigen():
    R = int(input("Enter the number of rows:"))
    C = int(input("Enter the number of columns:"))

    # Initialize matrix
    matrix = []
    print("Enter the entries rowwise:")

    # For user input
    for i in range(R):		 # A for loop for row entries
        a =[]
        for j in range(C):	 # A for loop for column entries
            a.append(int(input()))
        matrix.append(a)

    # For printing the matrix
    for i in range(R):
        for j in range(C):
            print(matrix[i][j], end = " ")
        print()

    n_array = np.array(matrix)
    w, v = np.linalg.eig(n_array)

    # printing eigen values
    print("Eigen values of the given square array:",w)
    # printing eigen vectors
    print("Right eigenvectors of the given square array:")
    print(v)


# In[77]:


find_eigen()


# In[84]:


def find_mean():
    # creating an empty list
    lst = []

    # number of elements as input
    n = int(input("Enter number of elements : "))

    # iterating till the range
    for i in range(0, n):
        ele = int(input("Enter element: "))

        lst.append(ele) # adding the element

    print("List given: ",lst)
    s_np = np.array(lst)
    print("Mean of the given list",s_np.mean())


# In[85]:


find_mean()


# In[93]:


def find_median():
    # creating an empty list
    lst = []
    # number of elements as input
    n = int(input("Enter number of elements : "))
    # iterating till the range
    for i in range(0, n):
        ele = int(input("Enter element: "))
        lst.append(ele) # adding the element
    print("List given: ",lst)
    print("Median of the given list",np.median(lst))


# In[94]:


find_median()


# In[185]:


def find_mode():
    from scipy import stats
    lst = []
    n = int(input("Enter number of elements : "))
    # iterating till the range
    for i in range(0, n):
        ele = int(input("Enter element: "))
        lst.append(ele) # adding the element
    print("List given: ",lst)
    print("Mode of the given list is: ",int(stats.mode(lst)[0]))


# In[186]:


find_mode()


# In[95]:


def find_stddev():
    # creating an empty list
    lst = []
    # number of elements as input
    n = int(input("Enter number of elements : "))
    # iterating till the range
    for i in range(0, n):
        ele = int(input("Enter element: "))
        lst.append(ele) # adding the element
    print("List given: ",lst)
    print("Standard Deviation of the given list",np.std(lst))


# In[97]:


find_stddev()


# In[218]:


def linreg():

    # def linreg():
    lst = []
    # number of elements as input
    print("Enter first list")
    n = int(input("Enter number of elements only Even : "))
    # iterating till the range
    for i in range(0, n):
        ele = int(input("Enter element: "))
        lst.append(ele) # adding the element
    print("List given: ",lst)
    X = np.array(lst)  # values converts it into a numpy array
    X = X.reshape(-1, 1)

    lstt = []
    print("Enter second list")
    n1 = int(input("Enter number of elements only Even : "))
    # iterating till the range
    for i in range(0, n1):
        elem = int(input("Enter element: "))
        lstt.append(elem) # adding the element
    print("List given: ",lstt)

    Y = np.array(lstt)
    Y = Y.reshape(-1, 1)

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(Y)  # make predictions
    Y_pred = Y_pred.reshape(-1,1)
    print(Y_pred)
    r_sq = linear_regressor.score(X,Y)
    print('coefficient of determination:', r_sq)

    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()


# In[219]:


linreg()


# In[220]:


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt  # To visualize
def logreg():
    lst = []
    # number of elements as input
    print("Enter first list")
    n = int(input("Enter number of elements only Even : "))
    # iterating till the range
    for i in range(0, n):
        ele = int(input("Enter element: "))
        lst.append(ele) # adding the element
    print("List given: ",lst)
    X = np.array(lst)  # values converts it into a numpy array
    X = X.reshape(-1, 1)

    lstt = []
    print("Enter second list")
    n1 = int(input("Enter number of elements only Even : "))
    # iterating till the range
    for i in range(0, n1):
        elem = int(input("Enter element: "))
        lstt.append(elem) # adding the element
    print("List given: ",lstt)

    Y = np.array(lstt)
    Y = Y.reshape(-1, 1)

    linear_regressor = LogisticRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(Y)  # make predictions
    Y_pred = Y_pred.reshape(-1,1)
    print(Y_pred)
    r_sq = linear_regressor.score(X,Y)
    print('coefficient of determination:', r_sq)
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()


# In[222]:


logreg()


# In[137]:


def add_matrix():
    R = int(input("Enter the number of rows:"))
    C = int(input("Enter the number of columns:"))

    # Initialize matrix
    matrix = []
    print("Enter the entries rowwise:")

    # For user input
    for i in range(R):		 # A for loop for row entries
        a =[]
        for j in range(C):	 # A for loop for column entries
            a.append(int(input("Enter Element: ")))
        matrix.append(a)

    # For printing the matrix
    for i in range(R):
        for j in range(C):
            print(matrix[i][j], end = " ")
        print()

    print("Enter Second Matrix")
    R1 = int(input("Enter the number of rows:"))
    C1 = int(input("Enter the number of columns:"))

    # Initialize matrix
    matrix1 = []
    print("Enter the entries rowwise:")

    # For user input
    for i in range(R1):		 # A for loop for row entries
        a1 =[]
        for j in range(C1):	 # A for loop for column entries
            a1.append(int(input("Enter Element:")))
        matrix1.append(a)

    # For printing the matrix
    for i in range(R1):
        for j in range(C1):
            print(matrix1[i][j], end = " ")
        print()

    result = [[matrix[i][j] + matrix1[i][j] for j in range(len(matrix[0]))] for i in range(len(X))]

#     for r in result:
    print("Result: ")
    print(r)


# In[138]:


add_matrix()


# In[139]:


def sub_matrix():
    R = int(input("Enter the number of rows:"))
    C = int(input("Enter the number of columns:"))

    # Initialize matrix
    matrix = []
    print("Enter the entries rowwise:")

    # For user input
    for i in range(R):		 # A for loop for row entries
        a =[]
        for j in range(C):	 # A for loop for column entries
            a.append(int(input("Enter Element: ")))
        matrix.append(a)

    # For printing the matrix
    for i in range(R):
        for j in range(C):
            print(matrix[i][j], end = " ")
        print()

    print("Enter Second Matrix")
    R1 = int(input("Enter the number of rows:"))
    C1 = int(input("Enter the number of columns:"))

    # Initialize matrix
    matrix1 = []
    print("Enter the entries rowwise:")

    # For user input
    for i in range(R1):		 # A for loop for row entries
        a1 =[]
        for j in range(C1):	 # A for loop for column entries
            a1.append(int(input("Enter Element:")))
        matrix1.append(a)

    # For printing the matrix
    for i in range(R1):
        for j in range(C1):
            print(matrix1[i][j], end = " ")
        print()

    result = [[matrix[i][j] - matrix1[i][j] for j in range(len(matrix[0]))] for i in range(len(X))]

#     for r in result:
    print("Result: ")
    print(r)


# In[140]:


def mult_matrix():
    R = int(input("Enter the number of rows:"))
    C = int(input("Enter the number of columns:"))

    # Initialize matrix
    matrix = []
    print("Enter the entries rowwise:")

    # For user input
    for i in range(R):		 # A for loop for row entries
        a =[]
        for j in range(C):	 # A for loop for column entries
            a.append(int(input("Enter Element: ")))
        matrix.append(a)

    # For printing the matrix
    for i in range(R):
        for j in range(C):
            print(matrix[i][j], end = " ")
        print()

    print("Enter Second Matrix")
    R1 = int(input("Enter the number of rows:"))
    C1 = int(input("Enter the number of columns:"))

    # Initialize matrix
    matrix1 = []
    print("Enter the entries rowwise:")

    # For user input
    for i in range(R1):		 # A for loop for row entries
        a1 =[]
        for j in range(C1):	 # A for loop for column entries
            a1.append(int(input("Enter Element:")))
        matrix1.append(a)

    # For printing the matrix
    for i in range(R1):
        for j in range(C1):
            print(matrix1[i][j], end = " ")
        print()

    result = [[matrix[i][j] * matrix1[i][j] for j in range(len(matrix[0]))] for i in range(len(X))]

#     for r in result:
    print("Result: ")
    print(r)


# In[150]:


def find_lim():
    from sympy import limit, oo, Symbol
    x = Symbol('x')
    xto = input("Enter x-> :")
    y = input("Enter function: ")
    ans = limit(y, x, xto)
    if ans == oo:
        print('Infinity')
    print(ans)


# In[152]:


find_lim()


# In[211]:


def expo():
    num = input("Enter number: ")
    num = float(num)
    res= np.format_float_positional(num, trim='-')
    print("Number in readable form is:",res)


# In[209]:


expo()


# In[3]:


def integral():
    num = input("Enter number: ")
    num = float(num)
    print("Integral of the given number is: ",sym.integrate(integral(num), (num, 0, 5)))
    #https://www.geeksforgeeks.org/how-to-find-definite-integral-using-python/


# In[4]:


integral()


# In[ ]:


print("Welcome!")
print("Choose the operation")
print("1. Logarithm")
print("2. Limits")
print("3. Mean")
print("4. Median")
print("5. Mode")
print("6. Standard Deviation")
print("7. Linear Regression")
print("8. Logistic Regression")
print("9. Matrices Addition")
print("10. Matrices Subraction")
print("11. Matrices Multiplication")
print("12. Determinants")
print("13. Eigen values And Eigen Vectors")
print("14. Not readable numbers? In e type? Try this out")
print("15. Integral")
print("Enter 0 to exit")
choice = input("Enter the choice(0-14): ")
choice = int(choice)
print("You chose: ",choice)
if choice == 0:
    print("Calculator closed")
if choice==1:
    find_log_with_base()
if choice==2:
    find_lim()
if choice==3:
    find_mean()
if choice==4:
    find_median()
if choice==5:
    find_mode()
if choice==6:
    find_stddev()
if choice==7:
    linreg()
if choice==8:
    logreg()
if choice==9:
    add_matrix()
if choice==10:
    sub_matrix()
if choice==11:
    mult_matrix()
if choice==12:
    find_matrix()
if choice==13:
    find_eigen()
if choice==14:
    expo()
if choice==15:
    integral()


# In[ ]:





# In[ ]:



# In[ ]:
