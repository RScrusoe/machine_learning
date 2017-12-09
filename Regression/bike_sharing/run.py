from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv('hour.csv',header=None)

def linear_regression_line(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) / 
        (mean(xs)**2 - mean(xs**2)))
    c = mean(ys) - m*mean(xs)
    return m,c

def square_error(y_orig, y_reg):
    return sum((y_orig-y_reg)**2)

def confidance(y_orig, y_reg):
    y_mean_line = [mean(y_orig) for y in y_orig]
    square_error_regression = square_error(y_orig,y_reg)
    square_error_y_mean = square_error(y_orig,y_mean_line)
    return 1 - (square_error_regression/square_error_y_mean)


# xs = np.array(d[0], dtype=np.float64)
# ys = np.array(d[1], dtype=np.float64)

xs = np.array([i for i in range(24)])
ys = np.array(d[2], dtype=np.float64)
a = [0 for i in range(24)]
while(len(ys)>0):
    t = ys[:24]
    a += t
    ys = ys[24:]
ys = a


#Dividing data in two sets
x1  = xs[:12]
x2 = xs[12:]

y1 = ys [:12]
y2 = ys[12:]
print(x1)
print(y1)
print(x2)
print(y2)

m,c = linear_regression_line(x1,y1)
reg_line = [m*x+c for x in x1]

confi_r = confidance(y1,reg_line) 
print(confi_r)
plt.scatter(x1,y1)
plt.plot(x1,reg_line)


m,c = linear_regression_line(x2,y2)
reg_line = [m*x+c for x in x2]

confi_r = confidance(y2,reg_line) 
print(confi_r)
plt.scatter(x2,y2)
plt.plot(x2,reg_line)

plt.show()