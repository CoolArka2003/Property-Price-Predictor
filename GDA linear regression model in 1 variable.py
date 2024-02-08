import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(r"C:\Users\CoolA\OneDrive\Documents\Code\deeplearning.mplstyle")

##################################################################
#The function compute_model_output calculates the sum of f_w,b(x_i)
def compute_model_output(x,w,b):
    '''
    Computes the gradient of the cost function
    Arguments:
        x (ndarray (m,)): Data, m examples
        w,b (scalar)    : model parameters
    Returns:
        f_wb: The sum of all values of f_wb(x_i)
    '''
    m=x.shape[0]
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i]+b
    return f_wb

##################################################################
#Generating the value of the cost function j_wb
def compute_cost(x,y,w,b):
    '''
    Computes the cost function
    Arguments:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target examples
        w,b (scalar)    : model parameters
    Returns:
        j_wb (scalar): The value of the cost function
    '''
    # generate the number of training examples
    m=x.shape[0]
    cost_sum = 0
    
    for i in range(m):
        f_wb=w*x[i]+b
        cost=(f_wb-y[i]) ** 2
        cost_sum=cost_sum+cost
    j_wb=((1/2*m))*cost_sum
    
    return j_wb

##################################################################
#compute_gradient calculates the value of the gradient dj_wb/dw and dj_wb/b
def compute_gradient(x,y,w,b):
    '''
    Computes the gradient of the cost function
    Arguments:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target examples
        w,b (scalar)    : model parameters
    Returns:
        dj_dw (scalar): The gradient of the cost w.r.t. parameter w
        dj_db (scalar): The gradient of the cost w.r.t. parameter b
        '''
    # generate the number of training examples
    m=x.shape[0]
    gradient_sum_dj_dw=0
    gradient_sum_dj_db=0

    for i in range(m):
        f_wb=w*x[i]+b
        gradient_sum_dj_dw+=(f_wb-y[i])*x[i]

    dj_dw=(1/m)*gradient_sum_dj_dw

    for i in range(m):
        f_wb=w*x[i]+b
        gradient_sum_dj_db+=(f_wb-y[i])

    dj_db=(1/m)*gradient_sum_dj_db

    return dj_dw, dj_db

##################################################################
# gradient_descent() auto calculates the value of w and b parameters
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    '''
    Auto computes the values of w and b parameters w.r.t the formula of Gradient Descent Algorithm
    Updates w,b by taking num_iters gradient steps with learning rate alpha

    Arguments:
        x (ndarray (m,))    : Data, m examples
        y (ndarray (m,))    : target examples
        w_in,b_in (scalar)  : initial values of model parameters
        alpha (float)       : learning rate
        num_iters (int)     : number of iterations to run gradient descent
        cost_function       : to call the value of the cost function J(w,b)
        gradient_function   : to call the value of the gradient of the cost function dj/dw and dj/db
    Returns:
        w (scalar)          : final value of parameter w
        b (scalar)          : final value of parameter b
        J_history (List)    : History of cost values
        p_history (List)    : History of parameters [w,b]
    '''
    
    #Declaring arrays to store the value of each iteration for graphing the data
    J_history = [] #history of cost values
    p_history = [] #history of parameters [w,b]
    b=b_in #setting b = initial value of b before implementation of GDA begins
    w=w_in #setting w = initial value of w before implementation of GDA begins

    # The code will be repeated until convergence, in code until num_iters
    for i in range(num_iters):
        
        # formula for gradient descent parameters is w = w - alpha*gradient of j_wb w.r.t. w and b = b - alpha*gradient of j_wb w.r.t. b
        # let's start with importing the value of gradient of j_wb
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # let's plug these values into the equation next
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # We will add these values to the history arrays now
        # We won't run this code millions of times over since that will exhaust the resources. Capping it to 100_000
        if i<100_000:
            J_history.append(compute_cost(x,y,w,b))
            p_history.append([w,b])
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
            
    return w,b,J_history,p_history

##################################################################
# the size of the properties in thousands of sq ft
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
#y_train stores the price of the properties in thousands of dollars
y_train = np.array([250, 300, 480,  430,   630, 730,])

'''w = 11.7016666666666666666666
b = 4.7

#Creating an object of the compute_model_output function
tmp_f_wb = compute_model_output(x_train, w,b,)



error=compute_cost(x_train,y_train,w,b,)
print(f"Error is ${error:.2f} thousand dollars")'''

#################################################################
#GDA implementation settings
w_init=0
b_init=0
num_iters=10000
tmp_alpha=1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train,y_train,w_init,b_init,tmp_alpha,num_iters,compute_cost,compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:9.4f},{b_final:8.4f})")

x_i=input("Enter the built area of your property in sq ft ")
cost=w_final*(float(x_i)/1000)+b_final
print(f"${cost} thousand dollars")

x_i=input("Enter the built area of your property in sq ft ")
cost=w_final*(float(x_i)/1000)+b_final
print(f"${cost} thousand dollars")

##################################################################
#Plotting the computer model
'''plt.plot(x_train,tmp_f_wb,c='b',label='Our Prediction')
plt.scatter(x_train,y_train,marker="x",c='r',label='Actual values')
plt.title("Housing prices")
plt.xlabel("Size (in 1000 sq ft)")
plt.ylabel("Price (in 1000s of dollars)")
plt.legend()
plt.show()'''
fig, (ax1,ax2) = plt.subplots(1,2,constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000+np.arange(len(J_hist[1000:])),J_hist[1000:])
ax1.set_title("Cost vs. Iterations (Start)");   ax2.set_title("Cost vs. Iteration (End)")
ax1.set_ylabel("Cost")                       ;   ax2.set_ylabel("Cost")
ax1.set_xlabel("iteration step")            ;   ax2.set_xlabel("iteration step")
plt.show()