import numpy as np;
import matplotlib.pyplot as plt;
import scipy.optimize as opt;

x = np.arange(1,16,1);
num = [4.00, 5.20, 5.900, 6.80, 7.34,
       8.57, 9.86, 10.12, 12.56, 14.32,
       15.42, 16.50, 18.92, 19.58, 20.00];
y = np.array(num);

def poly_fit():

    print(x);
    f1 = np.polyfit(x,y,3);
    p1 = np.poly1d(f1);

    print(p1);

    yvals = p1(x);

    plot1 = plt.plot(x,y,'s',label = 'original values');
    plot2 = plt.plot(x,yvals,'r', label = 'polyfit values');
    plt.xlabel('x');
    plt.ylabel('y');
    plt.legend(loc = 4);
    plt.title('polyfitting');
    plt.savefig('/Users/penpen926/workspace/MachineLearningPlayground/data/fit.png');

def func_to_fit(x, a,b,c):
    return a*np.exp(x/b)+c;

def fit_curve():
    # Fit the curv
    coeffs, cov = opt.curve_fit(func_to_fit, x,y);
    print(coeffs);
    print(cov);

    yvals = func_to_fit(x, coeffs[0], coeffs[1], coeffs[2]);
    plot1 = plt.plot(x,y,'s',label = 'orginal values');
    plot2 = plt.plot(x,yvals,'r', label = 'curve fitting values');
    plt.xlabel('x');
    plt.ylabel('y');
    plt.legend(loc = 4);
    plt.title("curve fitting");
    plt.savefig("/Users/penpen926/workspace/MachineLearningPlayground/data/curve_fit.png");

#poly_fit();
fit_curve();