import numpy as np
import matplotlib.pyplot as plt
plt.style.use(r"C:\Users\CoolA\Downloads\deeplearning.mplstyle")

#The function compute_model_output calculates the summation of f_w,b(x_i)
def compute_model_output(x,w,b):
    m=x.shape[0]
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i]+b
    return f_wb

#x_train stores the size of the properties in thousands of sq ft
x_train = np.array([1.0,2.0,3.5,4.0,6.5])
#y_train stores the price of the properties in thousands of dollars
y_train = np.array([300.0, 400.0, 470.0, 650.0, 750.0])

w = 85
b = 225

#Creating an object of the compute_model_output function
tmp_f_wb = compute_model_output(x_train, w,b,)

x_i=input("Enter the built area of your portland in Portland, OR ")
cost_1200sqft=w*float(x_i)/1000+b
print(f"${cost_1200sqft} thousand dollars")

#Plotting the computer model
plt.plot(x_train,tmp_f_wb,c='b',label='Our Prediction')
plt.scatter(x_train,y_train,marker="x",c='r',label='Actual values')
plt.title("Housing prices")
plt.xlabel("Size (in 1000 sq ft)")
plt.ylabel("Price (in 1000s of dollars)")
plt.legend()
plt.show()