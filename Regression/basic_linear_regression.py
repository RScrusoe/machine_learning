from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)
# xs=np.array([12,13,14,15,16,17,18,19,20,21,22,23])
# ys=np.array([202031,208153,188749,165747,150677,142722,136637,143456,135291,118952,103986,93421])

def linear_regression_line(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) / 
        (mean(xs)**2 - mean(xs**2)))
    c = mean(ys) - m*mean(xs)
    return m,c
m,c = linear_regression_line(xs,ys)
reg_line = [m*x+c for x in xs]

def square_error(y_orig, y_reg):
    return sum((y_orig-y_reg)**2)

def confidance(y_orig, y_reg):
    y_mean_line = [mean(y_orig) for y in y_orig]
    square_error_regression = square_error(y_orig,y_reg)
    square_error_y_mean = square_error(y_orig,y_mean_line)
    return 1 - (square_error_regression/square_error_y_mean)

confi_r = confidance(ys,reg_line) 
print(confi_r)
plt.scatter(xs,ys)
plt.plot(xs,reg_line)
plt.show()
