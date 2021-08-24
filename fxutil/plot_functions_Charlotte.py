from textwrap import wrap
from scipy import stats
import sqlite3
from matplotlib.lines import Line2D
from scipy.stats import norm, lognorm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits import mplot3d
import numpy as np, math, os
import read_store as sf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import collections as mc
import tumbling_busses_numba as tbn

from scipy import interpolate
from matplotlib import gridspec
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
from string import ascii_lowercase

# fitfunctions
def line_0(x, a):
    """quadratic function with b=0 and c=1 (has to go through (0,1)"""
    return (a * x)
def line(x, a,b):
    """quadratic function with b=0 and c=1 (has to go through (0,1)"""
    return (a * x)+b
def quadratic_ratio(x, a):
    """quadratic function with b=0 and c=1 (has to go through (0,1)"""
    return a * (x)**2 + 1

def quadratic_incr(x, a):
    """quadratic function with b=0 and c=1 (has to go through (0,1)"""
    return a * (x)**2

def quadratic_inverse(x, a, b):
    return a/x**2+b

def cubic(x, a, b, c, d):
    """cubic function"""
    return a * (x)**3 + b*x**2 + c*x + d

def cubic2(x, a, b,c):
    """cubic function through 0,0"""
    return a * (x)**3  + b*x**2+ c*x

def quintic(x, a, b,c,d,e,f):
    """quartic function through 0,0"""
    return a * (x)**5  + b*x**4+ c*x**3 + d*x**2 +e*x+f
def quartic(x, a, b,c,d):
    """quartic function through 0,0"""
    return a * (x)**4  + b*x**3+ c*x**2 + d*x
def quartic2(x, a, b):
    """quartic function through 0,0"""
    return a * (x)**4  + b*x**2

def expo_incr(x, a,b):
    """exponential function"""
    return a * (x)**b

def logarithmic_function(x, a,b):
    """logarithmic function"""
    return a * np.log(x)/math.log(b)

def exponential_decay(x, a,b):
    """exponential function with negative exponent"""
    return a * np.exp(-1*b*x)

def exponential_decay_log(x, a,b):
    """exponential function with negative exponent"""
    print(a,b)
    return math.log(a) * (b/x)

def exponential_decay_2(x, a,b):
    """exponential function with b as basis and a as factor"""
    return a * b**(-x)

def exponential_exp(x, a,b,c):
    """double exponential function with negative exponent"""
    return np.exp(a * np.exp(-1*b*x)+c)

def gompertz(x, alpha, beta, gamma, delta):
    return delta - alpha * np.exp(-1*beta*np.exp(-1*gamma * x))

def exponential_increase(x, a,b):
    """quadratic function with b=0 and c=1 (has to go through (0,1)"""
    return a * b**(x)

def exponential_saturation(x, a,b,c):
    """saturation function with exponential increase (has to go through (0,0)"""
    #return a/(1+ c* np.exp(-b*x))
    return a-b*np.exp(-x/c)

def exponential_shifted(x, a,b,c):
    """quadratic function with b=0 and c=1 (has to go through (0,1)"""
    return a * b**(-x)+c


def inverse_shifted(x, a,b,c):
    """quadratic function with b=0 and c=1 (has to go through (0,1)"""
    return a *b**x+c

def expo_incr_shifted(x, a,b,c,d):
    """quadratic function with b=0 and c=1 (has to go through (0,1)"""
    return a *((x-d)**b)+c

def quadratic_function(x, a, b, c):
    """quadratic function with b=-1 and c=1 (has to go through (1,1)"""
    return a * (x-b)**2 + c
def root_function(x, a,b, c):
    """quadratic function with b=0 and c=1 (has to go through (0,1)"""
    return b*(x)**(1/a)+c


def e_load_B(r, lmax, lam, B, v):
    """analytic function for effective load"""
    erg=(64 / 3 * r ** 5 / lmax ** 4 - 16 / 3  * r ** 3 / lmax ** 2 - 8 / 3 * r ** 2 / lmax + 2 / 3 * lmax)* lam/(v *B)
    return erg

def e_load(r, lmax, lam, x):
    """analytic function for effective load"""
    vB = 2 / 3 * lmax*lam/x
    erg=(64 / 3 * r ** 5 / lmax ** 4 - 16 / 3  * r ** 3 / lmax ** 2 - 8 / 3 * r ** 2 / lmax + 2 / 3 * lmax)* lam/(vB)
    return erg

def tanh(x,a):
    print(a)
    expo = 2*a*x
    temp = (np.exp(expo))
    ones = np.ones(len(x))
    res = temp+(-1)*ones
    res2=(temp+ones)
    return res/res2

def logistic(x,a,b,c):
    expo = (-1)*b*(x-c)
    res = a/(1+np.exp(expo))
    return res

def logistic_inverse(x,a,b,c):
    expo = (-1)*b*(x-c)
    res = a*(1-1/(1+np.exp(expo)))
    return res

def logistic_ratio(x,b):
    expo = (-1)*b*(x)
    res = 1/(1+np.exp(expo))
    return res

def my_function(x,alpha,b):
    """b = 1/l_max**2 from not served portion"""
    x2 = (x)**2
    #res = 1/((b*x2+1))-1/((a*4*x2+1))
    a = alpha*(0.5-2*x)**2
    res = (1-(4*b*x2))*a*x2/(1+a*x2)
    ##w_nss = (4*x2)*b
    #p = (1/(a*x2)+2+2*w_nss)
    #q=(2*w_nss-1-w_nss**2)*a*x2
    #res = (-1*p)/2 + np.sqrt(((p**2)/4)-q)
    return res

def my_function2(x,alpha,b, c):
    """b = 1/l_max**2 from not served portion"""
    x2 = (x)**2
    a = alpha*(0.5-2*x)**c
    res = (1-(4*b*x2))*a*x2/(1+a*x2)
    return res


def my_function3(x,alpha,b, c, d):
    """b = 1/l_max**2 from not served portion"""
    x2 = (x)**2
    a = alpha*(0.5-2*x)**4 + c*(0.5-2*x)**3 + d*(0.5-2*x)**2
    res = (1-(4*b*x2))*a*x2/(1+a*x2)
    return res

def my_function4(x,alpha,b, c, d):
    """b = 1/l_max**2 from not served portion"""
    x2 = (x)**2
    a = alpha*(0.5-2*x)**4 + c*(0.5-2*x)**2 + d
    res = (1-(4*b*x2))*a*x2/(1+a*x2)
    return res

def my_function5(x,alpha,b, c, d,e):
    """b = 1/l_max**2 from not served portion"""
    x2 = (x)**2
    a = alpha*(0.5-2*x)**5 + c*(0.5-2*x)**3 + d*(0.5-2*x)**2 +e
    res = (1-(4*b*x2))*a*x2/(1+a*x2)
    return res

def e_velocity_analytic(x, alpha, beta, gamma , lam, v, B, lmax, ts):
    """"calculates e vlocity with parameters alpha and beta"""
    wn= (2*x/lmax)**2
    res = beta*(wn**(1/alpha))+gamma
    return res

def e_velocity_analytic2(x, alpha, beta , lam, v, B, lmax, ts):
    """"calculates e vlocity with parameters alpha and beta"""
    z=(1-2*x/lmax)**2
    v0 =alpha*ts
    y=(64/(3*lmax**4) * x ** 5 - 16/(3*lmax**2) *x** 3 - 8/(3*lmax)* x ** 2 +2/3*lmax)*lam/B*(lmax-2*x)**2*x**2
    p = (y+z*beta/ts-beta/ts-v0)
    q = -(y*beta)/ts-v0*y
    res = -p/2+np.sqrt(p**2/4-q)
    return res

def directly_served_analytic(x, alpha, lam, v, B, lmax):
    b = 1 / lmax ** 2
    x2 = (x) ** 2
    a = (alpha)* ((64/(3*lmax**4) * x ** 5 - 16/(3*lmax**2) *x** 3 - 8/(3*lmax)* x ** 2 +
                                   2/3*lmax)*lam/(v*B))**2
    res = (1 - (4 * b * x2)) *(1 / (1 + a * x2))
    return res

def directly_served_analytic2(x, alpha, lam, v, B, lmax):
    b = 1 / lmax ** 2
    x2 = (x) ** 2
    a = (alpha)* (lmax-2*x)**2*((64/(3*lmax**4) * x ** 5 - 16/(3*lmax**2) *x** 3 - 8/(3*lmax)* x ** 2 +
                                   2/3*lmax)*lam/(v*B))
    res = (1 - (4 * b * x2)) *(1 / (1 + a * x2))
    return res

def indirectly_served_analytic2(x, alpha, lam, v, B, lmax):
    b = 1 / lmax ** 2
    x2 = (x) ** 2
    a = (alpha)* ((64/(3*lmax**4) * x ** 5 - 16/(3*lmax**2) *x** 3 - 8/(3*lmax)* x ** 2 +
                                   2/3*lmax)*lam/(v*B))**2
    res = (1 - (4 * b * x2)) *(1-1 / (1 + a * x2))
    return res

def indirectly_served_analytic(x, alpha, lam, v, B, lmax):
    b = 1 / lmax ** 2
    x2 = (x) ** 2
    a = (alpha)*(lmax-2*x)**2*((64/(3*lmax**4) * x ** 5 - 16/(3*lmax**2) *x** 3 - 8/(3*lmax)* x ** 2 +
                                   2/3*lmax)*lam/(v*B))
    res = (1 - (4 * b * x2)) *(1-1 / (1 + a * x2))
    return res


def indirectly_served_analyticve(x, alpha, lam, v, B, lmax, ts):
    a=14.526329	*ts+5.526976
    b=5.179272		*ts+0.155065
    c=1.033836*np.exp(-25.00833*ts)
    v_e = b * (x ** 2) ** (1 / a)+c

    b = 1 / lmax ** 2
    x2 = (x) ** 2
    a = (alpha)*(lmax-2*x)**2*((64/(3*lmax**4) * x ** 5 - 16/(3*lmax**2) *x** 3 - 8/(3*lmax)* x ** 2 +
                                   2/3*lmax)*lam/(v_e*B))
    res = (1 - (4 * b * x2)) *(1-1 / (1 + a * x2))
    return res


def indirectly_served_analytic3(x, alpha, lam, v, B, lmax):
    x=x/4
    b = 1 / lmax ** 2
    x2 = (x) ** 2
    a = (alpha)* ((64/(3*lmax**4) * x ** 5 - 16/(3*lmax**2) *x** 3 - 8/(3*lmax)* x ** 2 +
                                   2/3*lmax)*lam/(v*B))**2
    res = (1 - (4 * b * x2)) *(1-1 / (1 + a * x2))
    return res

def indirectly_served_analytic4(x, alpha, lam, v, B, lmax):
    x=x/4
    b = 1 / lmax ** 2
    x2 = (x) ** 2
    a = (alpha)*(lmax-2*x)**2*((64/(3*lmax**4) * x ** 5 - 16/(3*lmax**2) *x** 3 - 8/(3*lmax)* x ** 2 +
                                   2/3*lmax)*lam/(v*B))
    res = (1 - (4 * b * x2)) *(1-1 / (1 + a * x2))
    return res

def indirectly_served_analytic_ql(x, alpha, lmax, ql):
    b = 1 / lmax ** 2
    x2 = (x) ** 2
    a = alpha* ql
    res = (1 - (4 * b * x2)) * a * x2 / (1 + a * x2)
    return res

def not_served_analytic(r, n=None):
    """not served portion for square lattice with finite size n
    input: r = relative pool radius in [0,1]
    (half of distance that can be walked relative to maximal stop distance"""
    if n==None:
        n = 50000
    # drel_arr = np.linspace(0,1,dmax)
    erg=np.zeros(len(r))
    d_arr = r
    for counter in range(len(d_arr)):
        d= d_arr[counter]
        if 0.5>=d:
            erg[counter]= (4 *(-1 + n) *(d**4 *(-1 + n)**2 - 4 *d**3 *(-1 + n)* n + (d**2* (-1 + 6 *n**2))/2))/3
        elif d<=2*(n-1):
            erg[counter]=  (4 *(-1 + n) *(0.5**4 *(-1 + n)**2 - 4 *0.5**3 *(-1 + n)* n + (0.5**2 *(-1 + 6 *n**2))/2))\
                           /3+\
                        (2* (d**2 *(-1 + n) - 2* d* n - (-2* d *(-1 + n) + 2* n)**4/(8* (-1 + n))))/3 -\
                (2* (0.5**2 *(-1 + n) - 2* 0.5* n - (-2* 0.5* (-1 + n) + 2* n)**4/(8* (-1 + n))))/3
        else:
            erg[counter]= None
    return erg*2*(n-1)/(n**2*(n**2-1))
    return res


def log_normal(x, sigma, mu):
    return 1/(sigma*x*math.sqrt(2*math.pi))*np.exp((-1)*(np.log(x)-mu)**2/(2*sigma**2))

def normal(x, mean, sd):
    var = sd**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(x-mean)**2/(2*var))
    return num/denom

def gaussian(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def set_fit_function(fitter, lam, v, B, lmax, fitted, last_label, last_popt, p0, ts, parameter_value,
                     ql=None):
    """selects fit function by fitter, returns function, fitlabel and p0"""
    function=None
    fitlabel=None
    if fitter == 'l' or fitter == 'line' or fitter == 1:
        function = line
        fitlabel = r'$y={a}\cdot x+{b}$'
    elif fitter == 'constant' or fitter == 1:
        function = lambda a,x: a
        fitlabel = r'$y={a}$'
    if fitter == 'line0':
        function = lambda x, a: line(x, a, 0)
        fitlabel = r'$y={a} \cdot {x}$'
    elif fitter == 'r' or fitter == 'root':
        # calculate and plot curve fits
        function = lambda x, a, b, c: root_function(x, a, b,c)
        fitlabel = r'$y={b} {x}^{1/{a}}+{c}$'
    elif fitter == 'q' or fitter == 'quadratic':
        # calculate and plot curve fits
        function = lambda x, a, b, c: quadratic_function(x, a, b, c)

        fitlabel = r'$y={a} ({x}-{b})^2+{c}$'
    elif fitter == 'q2' or fitter == 'quadratic2':
        # calculate and plot curve fits
        function = lambda x, a: quadratic_function(x, a, 0, 0.33)
        fitlabel = r'$y={a} {x}^2+0.33$'
    elif fitter == 'q3' or fitter == 'quadratic3':
        # calculate and plot curve fits
        function = lambda x, a, b: quadratic_function(x, a, 0, 0)
        fitlabel = r'$y={a} {x}^2$'
    elif fitter == 'cubic':
        # calculate and plot curve fits
        function = cubic
        fitlabel = r'$y={a} {x}^3 + {b} {x}^2+{c}{x}+{d}$'
    elif fitter == 'cubic2':
        # calculate and plot curve fits
        function = cubic2
        fitlabel = r'$y={a} {x}^3 + {b} {x}^2+{c}{x}$'
    elif fitter == 'cubicpure':
        # calculate and plot curve fits
        function = lambda x, a: cubic(x,a,0,0,0)
        fitlabel = r'$y={a} {x}^3 $'
    elif fitter == 'quartic':
        # calculate and plot curve fits
        function = quartic
        fitlabel = r'$y={a} {x}^4 + {b} {x}^3+{c} {x}^2+{d}{x} $'
    elif fitter == 'quartic2':
        # calculate and plot curve fits
        function = quartic2
        fitlabel = r'$y={a} {x}^4 + {b} {x}^2 $'
    elif fitter == 'quintic':
        # calculate and plot curve fits
        function = quintic
        fitlabel = r'$y={a} {x}^5 + {b} {x}^4+{c} {x}^3+{d}{x}^2+{e}{x}+{f} $'
    elif fitter == 'e' or fitter == 'exponential_decay':
        # calculate and plot curve fits
        function = exponential_decay
        fitlabel = r'$y={a} e^{-{b} {x}}$'
    elif fitter == 'ei' or fitter == 'exponential_increase':
        # calculate and plot curve fits
        function = lambda x, a, b: exponential_decay(x, a, -b)
        fitlabel = r'$y={a} e^{{b} {x}}$'
    elif fitter == 'e_log' or fitter == 'exponential_decay_log':
        # calculate and plot curve fits
        function = lambda x, a, b: exponential_decay_log(np.log(x), a, b)
        fitlabel = r'$y={a} e^{-{b} {x}}$'  # TODO does not work due to log of negative number
    elif fitter == 'e2' or fitter == 'exponential_decay_2':
        # calculate and plot curve fits
        function = exponential_decay_2
        fitlabel = r'$y={a} \cdot {b} ^{-{x}}$'
    elif fitter == 'esat' or fitter == 'exponential_saturation':
        # calculate and plot curve fits
        function = exponential_saturation
        fitlabel = r'$y={a}-{b} \cdot {e} ^{{x}/{c}}$'
    elif fitter == 'gompertz':
        # calculate and plot curve fits
        function = gompertz
        fitlabel = r'$y={d}-{a} \cdot e^{-{b}\cdot e ^{- {c}{x}}}$'
    elif fitter == 'gompertz0':
        # calculate and plot curve fits
        function = lambda x, b, c: gompertz(x,-1,b,c,0)
        fitlabel = r'$y=- \cdot e^{-{b}\cdot e ^{- {c}{x}}}$'
    elif fitter == 'logistic':
        # calculate and plot curve fits
        function = logistic
        fitlabel = r'$y=\frac{{a}}{1+e^{-{b}({x}-{c})}}$'
    elif fitter == 'logistic_inverse':
        # calculate and plot curve fits
        function = logistic_inverse
        fitlabel = r'$y=a\left(1-\frac{{1}}{1+e^{-{b}({x}-{c})}}\right)$'
    elif fitter == 'g' or fitter == 'gaussian':
        # calculate and plot curve fits
        function = lambda x, a, x0, sigma: gaussian(x, a, x0, sigma)
        fitlabel = r'$y={a} \cdot e^{-({x}-{b})/(2{c}^2)}$'
    elif fitter == 'es' or fitter == 'exponential_shifted':
        # calculate and plot curve fits
        function = lambda x, a, b, c: inverse_shifted(x, a, b, c)
        fitlabel = r'$y={a} \cdot {b} ^{x}+{c}$'
        if fitted:
            if last_label == fitlabel:
                p0 = last_popt
    elif fitter == 'es2' or fitter == 'exponential_shifted2':
        # calculate and plot curve fits
        function = lambda x, a, b: exponential_shifted(x, a, b, 0)
        fitlabel = r'$y={a} \cdot {b} ^{-{x}}$'
        if fitted:
            if last_label == fitlabel:
                p0 = last_popt
    elif fitter == 'es1' or fitter == 'exponential_shifted1':
        # calculate and plot curve fits
        function = lambda x, a: inverse_shifted(x, -0.1, a, 1)
        fitlabel = r'$y=-0.1 \cdot {a} ^{{x}}+1$'
        if fitted:
            if last_label == fitlabel:
                p0 = last_popt
    elif fitter == 'ee' or fitter == 'exponential_exp':
        # calculate and plot curve fits
        function = lambda x, a, b, c: exponential_exp(x, a, b, c)
        fitlabel = r'$y=\mathrm{exp}({a} \cdot {e} ^{-{b}{x}}+{c})$'
        if fitted:
            if last_label == fitlabel:
                p0 = last_popt
    elif fitter == 'i' or fitter == 'inverse_shifted':
        # calculate and plot curve fits
        function = lambda x, a, b, c: inverse_shifted(x, a, b, c)
        fitlabel = r'$y={a} /{b}^ {x}+{c}$'
        if fitted:
            if last_label == fitlabel:
                p0 = last_popt
    elif fitter == 'i2' or fitter == 'inverse_squared':
        # calculate and plot curve fits
        function = lambda x, a, b, c, d: expo_incr_shifted(x, a, -b, c, d)
        fitlabel = r'$y={a} \cdot ({x}-{d})^{-{b}}+{c}$'
        if fitted:
            if last_label == fitlabel:
                p0 = last_popt

    elif fitter == 't' or fitter == 'travel_time':
        # calculate and plot curve fits
        function = lambda x, a, b: expo_incr_shifted(x, a, -b, 1/3 , 0)
        fitlabel = r'$y={a}\cdot {x}^{-{b}}+1$'
        if fitted:
            if last_label == fitlabel:
                p0 = last_popt
    elif fitter == 't2' or fitter == 'rel_travel_time':
        # calculate and plot curve fits
        function = lambda x, a, b: expo_incr_shifted(x, a, -b, 1 , 0)
        fitlabel = r'$y={a}\cdot {x}^{-{b}}+1$'
        if fitted:
            if last_label == fitlabel:
                p0 = last_popt
    elif fitter == 'ti' or fitter == 'rel_travel_time_invers':
        # calculate and plot curve fits
        function = lambda x, a, b: ((x-1)/a)**(-1/b)
        fitlabel = r'$y={a}\cdot {x}^{-{b}}+1$'
        if fitted:
            if last_label == fitlabel:
                p0 = last_popt
    elif fitter == 't3' or fitter == 'rel_travel_time':
        # calculate and plot curve fits
        function = lambda x, a: quadratic_inverse(x, a, 1 )
        fitlabel = r'$y=\dfrac{{a}}{{x}^{2}}+1$'
        if fitted:
            if last_label == fitlabel:
                p0 = last_popt
    elif fitter == 'ip' or fitter == 'inverse_pure':
        # calculate and plot curve fits
        function = lambda x, a: inverse_shifted(x, a, 0)
        fitlabel = r'$y={a} /{x}$'
        if fitted:
            if last_label == fitlabel:
                p0 = last_popt
    elif fitter == 'log' or fitter == 'logarithmic':
        # calculate and plot curve fits
        function = logarithmic_function
        fitlabel = r'$y={a} \cdot log{x}/log{b}$'
        if fitted:
            if last_label == fitlabel:
                p0 = last_popt
    elif fitter == "my_function":
        b = 1 / 0.5 ** 2
        function = lambda x, a: my_function(x, a, b)
        precision = None
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot a\cdot(l-2r)^2\cdot r^2}{(1+a\cdot(l-2r)^2\cdot r^2)}, a={a}$'
    elif fitter == "my_function2":
        b = 1 / 0.5 ** 2
        c = 2.3
        function = lambda x, a: my_function2(x, a, b, c)
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot a\cdot(l-2r)^c\cdot r^2}{(1+a\cdot(l-2r)^c\cdot r^2)}, ' \
                   r'a={a}, c={c}$'
    elif fitter == "m" or fitter == "my_function3":
        b = 1 / 0.5 ** 2
        function = lambda x, a, c, d: my_function3(x, a, b, c, d)
        precision = None
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot \alpha(l-2r)\cdot r^2}{(1+\alpha(l-2r)\cdot r^2)}$' + '\n' + \
                   r'$\alpha({x}) = {a}{x}^4+{b}{x}^3+{c}{x}^2$'
    elif fitter == "my_function4":
        b = 1 / 0.5 ** 2
        function = lambda x, a, c, d: my_function4(x, a, b, c, d)
        precision = None
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot \alpha(l-2r)\cdot r^2}{(1+\alpha(l-2r)\cdot r^2)}$' + '\n' + \
                   r'$\alpha({x}) = {a}{x}^4+{b}{x}^2+{c}$'
    elif fitter == "dir":
        print(lam, v, B, lmax)
        function = lambda x, a: directly_served_analytic(x, a, lam, v, B, lmax)
        precision = None
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot \alpha(l-2r)\cdot r^2}{(1+\alpha(l-2r)\cdot r^2)}$' + '\n' + \
                   r'$\alpha(l-2r)={a}\cdot x_\mathrm{e}^2(r)$'
    elif fitter == "dir2":
        print(lam, v, B, lmax)
        function = lambda x, a: directly_served_analytic2(x, a, lam, v, B, lmax)
        precision = None
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot \alpha(r)\cdot r^2}{(1+\alpha(r)\cdot r^2)}$' + '\n' + \
                   r'$\alpha(r)={a}\cdot (l-2r)^2\cdot x_\mathrm{e}(r)$'
    elif fitter == "ind2":
        print(lam, v, B, lmax)
        function = lambda x, a: indirectly_served_analytic2(x, a, lam, v, B, lmax)
        precision = None
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot \alpha(r)\cdot r^2}{(1+\alpha(r)\cdot r^2)}$' + '\n' + \
                   r'$\alpha(r)={a}\cdot x_\mathrm{e}^2(r)$'
    elif fitter == "rind2":
        print(lam, v, B, lmax)
        function = lambda x, a: indirectly_served_analytic2(x/4, a, lam, v, B, lmax)
        precision = None
        fitlabel = r'$y=\dfrac{(1-\tilde{r}^2)\cdot \alpha(\tilde{r})\cdot \tilde{r}^2}{(1+\alpha(\tilde{r})\cdot \tilde{r}^2)}$' + '\n' + \
                   r'$\alpha(\tilde{r})={a}\cdot \tilde{x}_\mathrm{e}^2(\tilde{r})$'
    elif fitter == "ind":
        print(lam, v, B, lmax)
        function = lambda x, a: indirectly_served_analytic(x, a, lam, v, B, lmax)
        precision = None
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot \alpha(r)\cdot r^2}{(1+\alpha(r)\cdot r^2)}$' + '\n' + \
                   r'$\alpha(r)={a}\cdot (l-2r)^2x_\mathrm{e}(r)$'
    elif fitter == "indve":
        # print(lam, v, B, lmax)
        function = lambda x, a: indirectly_served_analyticve(x, a, lam, v, B, lmax, ts)
        precision = None
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot \alpha(r)\cdot r^2}{(1+\alpha(r)\cdot r^2)}$' + '\n' + \
                   r'$\alpha(r)={a}\cdot (\ell_\mathrm{max}-2r)^2\cdot\frac{\langle \ell \rangle_\mathrm{e}(' \
                   r'r)\lambda_\mathrm{' \
                   r'e}(' \
                   r'r)}{(v_\mathrm{e}(r)\cdot B)}$'
    elif fitter == "indr":
        print(lam, v, B, lmax)
        function = lambda x, a: indirectly_served_analytic(x/4, a, lam, v, B, lmax)
        precision = None
        fitlabel = r'$y=\dfrac{1-2\tilde{r}+\tilde{r}^3-\tilde{r}^4}{1+{a}\cdot (1-\tilde{r})^2 x' \
                   r'\left( \tilde{r} ^7-\tilde{r} ^5-\tilde{r} 4+\tilde{r}^2\right) }$'
    elif fitter == "ind3": # for \tilde{r} instead of r
        print(lam, v, B, lmax)
        function = lambda x, a: indirectly_served_analytic3(x, a, lam, v, B, lmax)
        precision = None
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot \alpha(r)\cdot r^2}{(1+\alpha(r)\cdot r^2)}$' + '\n' + \
                   r'$\alpha(r)={a}\cdot x_\mathrm{e}^2(r)$'
    elif fitter == "ind4":# for \tilde{r} instead of r
        print(lam, v, B, lmax)
        function = lambda x, a: indirectly_served_analytic4(x, a, lam, v, B, lmax)
        precision = None
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot \alpha(r)\cdot r^2}{(1+\alpha(r)\cdot r^2)}$' + '\n' + \
                   r'$\alpha(r)={a}\cdot (l-2r)^2x_\mathrm{e}(r)$'
    elif fitter == "indql":# for \tilde{r} instead of r
        function = lambda x, a: indirectly_served_analytic_ql(x, a, lmax,ql)
        precision = None
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot \alpha(r)\cdot r^2}{(1+\alpha(r)\cdot r^2)}$' + '\n' + \
                   r'$\alpha(r)={a}\cdot (l-2r)^2x_\mathrm{e}(r)$'
    elif fitter == "wn":# for \tilde{r} instead of r
        function = lambda x: not_served_analytic(x)
        precision = None
        fitlabel = r'$y=\Phi(2r)$'
    elif fitter == "ev":
        print(lam, v, B, lmax, ts)
        function = lambda x, a, b,c:e_velocity_analytic(x, a, b,c,lam, v, B, lmax, ts)
        precision = None
        fitlabel = r'$y={b}\left(\omega_\mathrm{n}(r)\right)^{1/{a}}+{c}$'
    elif fitter == "ev2":
        print(lam, v, B, lmax, ts)
        function = lambda x, a, b:e_velocity_analytic2(x, a, b,lam, v, B, lmax, ts)
        precision = None
        fitlabel = r'$y=\dfrac{(1-(2r)^2/l^2)\cdot \alpha(l-2r)\cdot r^2}{(1+\alpha(l-2r)\cdot r^2)}$' + '\n' + \
                   r'$\alpha(l-2r)={a}\cdot(l-2r)^2\cdot x_\mathrm{e}$'
    elif fitter == "e_load":
        function = lambda x, a: e_load(x, lmax, lam, parameter_value)
        fitlabel = r"$y=\dfrac{\left(\frac{64}{3}\frac{r^5}{l_{\mathrm{max}}^4}-\frac{16}{3} \frac{r^3}" \
                   r"{ l_{\mathrm{max}}^2}-\frac{8}{3}\frac{r^2}{l_{\mathrm{max}}}+\frac{2}{3} l_" \
                   r"{\mathrm{max}}\right)x  }{\frac{2}{3} l_{\mathrm{max}}}$" + \
                   '\n' + r'$l_{\mathrm{max}}=%5.1f$' % (lmax)
    elif fitter == "e_loadr":
        function = lambda x, a: e_load(x/4, lmax, lam, parameter_value)
        fitlabel = r"$y=\frac{2}{3} \ell_{\mathrm{" \
                   r"max}}\cdot x  \left(\tilde{r}^5-\tilde{r}^3-\tilde{r}^2+1\right)$" + \
                   '\n' + r'$\ell_{\mathrm{max}}=%5.1f$' % (lmax)
    elif fitter == "e_load_B":
        function = lambda x, a: e_load_B(x, lmax, lam, parameter_value, 1)
        fitlabel = r"$y=\dfrac{\left(\frac{64}{3}\frac{r^5}{l_{\mathrm{max}}^4}-\frac{16}{3} \frac{r^3}" \
                   r"{ l_{\mathrm{max}}^2}-\frac{8}{3}\frac{r^2}{l_{\mathrm{max}}}+\frac{2}{3} l_" \
                   r"{\mathrm{max}}\right)\lambda  }{v_\mathrm{v}B}$" + \
                   '\n' + r'$l_{\mathrm{max}}=%5.1f,\lambda\approx%5.0f,v_\mathrm{v}=%5.0f$' % (lmax, lam, 1)
    elif fitter == "e_load_Br":
        function = lambda x, a: e_load_B(x/4, lmax, lam, parameter_value, 1)
        fitlabel = r"$y=\dfrac{ \frac{2}{3} \ell_{\mathrm{max}}\lambda}{v_\mathrm{v}B} " \
                   r"\left(\tilde{r}^5-\tilde{r}^3-\tilde{r}^2+1\right)$"
                   # + '\n' + r'$\ell_{\mathrm{max}}=%5.1f,\lambda\approx%5.0f,v_\mathrm{v}=%5.0f$' % (lmax, lam, 1)
    return function, fitlabel, p0

#helpful functions
def calc_mean_with_error(input_array):
    """calculates mean and error for axis>1 where axis=0 is unique
    :param input array like [[group_by_this],[observable1],[observable2],...]
    :returns values like unique_group_by_vals, [[observable1_means],[observable2_means],...], [[observable1_errors],
    [observable2_errors],...]"""
    unique_pool_radius = np.unique(input_array[0])
    width,lenght = np.shape(input_array)
    #print(lenght, width)
    means = np.zeros((width-1,len(unique_pool_radius)))
    errs = np.zeros((width-1,len(unique_pool_radius)))
    for pool_radius_counter in range(len(unique_pool_radius)):
        pool_radius_value = unique_pool_radius[pool_radius_counter]
        #print('value',pool_radius_value )
        indices = np.where(input_array[0]==pool_radius_value)[0]
        for counter in range(1,width):
            #print(np.dtype(input_array[counter, indices]))
            #print(np.where(~np.isnan(input_array[counter, indices]))[0],np.where(np.isnan(input_array[counter,
            # indices]))[0])
            try:
                notnan_indices = np.where(~np.isnan(input_array[counter, indices]))[0]
                means[counter-1, pool_radius_counter] = np.mean(input_array[counter, indices][notnan_indices])
                errs[counter-1, pool_radius_counter] = stats.sem(input_array[counter, indices][notnan_indices], axis=None)
            except TypeError:
                means[counter - 1, pool_radius_counter] = np.mean(input_array[counter, indices])
                errs[counter - 1, pool_radius_counter] = stats.sem(input_array[counter, indices],
                                                                   axis=None)
    return unique_pool_radius, means, errs


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def add_discrete_cbar(normalize, colormap, unique_parameter, fig, small_font, middle_font, thin_linewidth,
                      linewidth, ylabel, axis = None, mappable=None,precision=1):
    """adds colorbar with colormap and normalization (discrete), ticks set for unique_pool_radius"""
    if mappable == None:
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappaple.set_array(unique_parameter)
        # print([(colormap(normalize(i)),normalize(i), i) for i in unique_parameter])
    else:
        scalarmappable = mappable
    # print('unique_parameter=np.array(',list(unique_parameter),')')
    bounds = np.array(np.concatenate((unique_parameter, np.array([unique_parameter[-1] *2 -
                                                                        unique_parameter[-2]]))),dtype=float)
    # print(bounds)
    bounddif = np.diff(bounds).mean()/2
    bounds -= bounddif
    if unique_parameter.max() <= 1:
        if len(unique_parameter)>10:
            indice_distance = int(len(unique_parameter)/5)
            indices = list(np.arange(5)*indice_distance)+[-1]
            cbar_ticks = unique_parameter[indices]
        else:
            cbar_ticks = unique_parameter
    else:
        if len(unique_parameter)>10:
            par_min = np.min(unique_parameter)
            par_max = np.max(unique_parameter)
            cbar_ticks=np.arange(par_min, par_max + 1, (par_max -par_min)/4, dtype=int )
        else:
            cbar_ticks = unique_parameter
    cbar_ticklabels = cbar_ticks.round(precision)
    # print(cbar_ticks)
    if axis==None:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    else:
        cbar_ax = axis
    format = None
    # print(round(bounds.max()),round(bounds.min()),bounds.max(),bounds.min())
    if bounds.min()>=0:
        # print("runden")
        for number in [5,4,3,2,1]:
            # print(cbar_ticklabels.round(number), cbar_ticklabels)
            if np.array_equal(cbar_ticklabels.round(number), cbar_ticklabels):
                format = '%.'+str(number)+'f'
        if np.array_equal(cbar_ticklabels.round(0),cbar_ticklabels) :
            format='%i'
    # print('unique_parameter',unique_parameter)
    # print('bounds',  bounds, np.diff(bounds))
    # print('cbar_ticks',cbar_ticks, unique_parameter, unique_parameter.max())
    # print('format',format)
    dif = np.diff(unique_parameter).round(5)
    # print('dif', dif, np.unique(dif))
    if len(np.unique(dif))>1:
        #bounds are not equally distant
        try:
            cbar=fig.colorbar(scalarmappaple, ticks=cbar_ticks, cax=cbar_ax, pad=0.2,format=format)
        except ValueError:
            print('ValueError')
            cbar=fig.colorbar(scalarmappaple,  cax=cbar_ax, pad=0.2)
    else:
        try:
            cbar=fig.colorbar(scalarmappaple, boundaries=bounds, ticks=cbar_ticks, cax=cbar_ax, pad=0.2,format=format)
        except ValueError:
            print('ValueError')
            cbar=fig.colorbar(scalarmappaple,  cax=cbar_ax, pad=0.2)
    cbar_ax.set_yticklabels(cbar_ticklabels)
    cbar_ax.tick_params(labelsize=small_font, width=thin_linewidth, length=linewidth)
    cbar_ax.set_ylabel(ylabel, {'fontsize': middle_font})
    # print('cbar ticks', list(cbar_ax.get_yticklabels()), list(cbar_ax.get_yticks()))
    return cbar

#used plot functions


def plot_anything(df_input,directory, plotname, fitter, variable, axislabel, value, valuelabel, parameter=None,
                  parameterstr=None, constant=None, title=None, yerr = None,xerr = None, yscale = "linear",xscale="linear",
                  cscale="linear",colormap = truncate_colormap(cm.Greens, minval=0.4, maxval=1) , color = None,
                  divide_by=None,multiply_by=None, analytic=None, mark_max = False,mark_min = False,cbar=False, xlim=None,
                  parameterset ='latin', precision=4, short_axis_label = None, short_value_label = None, ylim=None,
                  discrete = False, scientific_axis="",figsize=(6, 4.5),arrow=[],sum_legend=False,  point=[],legendpos=None,
                  all_fitters=True, p0_in=None,abstract_fit=True,vline=None, suppress_cbar=False,
                  suppress_legend=False,small_font=16,middle_font = 20,large_font = 22,markersize = 12,linewidth = 3,
                  thin_linewidth = 2,marker="o",secondaxis=[],maxlegendlen=5, bars=False, step=False):
    """plots value against variable for different parameters use different colors
     value has to be numeric or at least convertable to float
     can add constant value or analytic function, multiply or divide the value by a constant and set title by hand
     analytic and constand require format [values, color, label]
     input arrow: list of arrows to draw with [x1,y1,x2,y2,color]
    input point: list of points to draw with [x,y,text,percentual distance of text,color]
     title="" will not set a title, None will design a title from labels
     if sum_legend, clabel and fit will be unified
    """
    if short_axis_label == None:
        split_axislabel = axislabel.split('$')
        if len(split_axislabel)>1:
            short_axis_label = split_axislabel[-2]
    if short_value_label == None:
        split_valuelabel = valuelabel.split('$')
        if len(split_valuelabel)>1:
            short_value_label = split_valuelabel[-2]
    plt.rcdefaults()
    if fitter == 'connect' or fitter =='c':
        linestyle = "--"
    elif fitter == 'cc' or fitter =='-':
        linestyle = "-"
    else:
        linestyle = ''
    lmax = 0.5
    if "irpt" in df_input.columns:
        lam = df_input.irpt.iloc[0]
    else:
        lam = None
    if "stop_time" in df_input.columns:
        ts = df_input.stop_time.iloc[0]
    else:
        ts = None
    if "number_of_transporters" in df_input.columns:
        B = df_input.number_of_transporters.iloc[0]
    else:
        B = None
    v=1
    keep_columns = [variable, value]
    if xerr != None:
        keep_columns += [xerr]
    if yerr != None and yerr != "off":
        keep_columns += [yerr]
    if parameter != None:
        keep_columns += [parameter]
    # print(keep_columns)
    keep_columns=np.unique(np.array(keep_columns)) # if parameter=something else dont have column twice
    df = df_input[keep_columns].copy()
    df = df[~pd.isna(df[value])]
    df.loc[:,value] = df.loc[:,value].astype(float)
    pool_radius = df[variable].to_numpy()
    df.loc[:, "value"] = df[value]
    if divide_by!=None:
        df.loc[:,"value"] = df.value/df[divide_by]
    if multiply_by!=None:
        df.loc[:,"value"] = df.value*df[multiply_by]
    value_data = df.value.to_numpy()
    if parameter != None:
        parameter_data = df[parameter].to_numpy()
    else:
        parameter_data = np.zeros(len(value_data))
    if xerr != None:
        xerr_data = df[xerr].to_numpy()
    else:
        xerr_data = np.zeros(len(value_data))
    if yerr != None and yerr != "off":
        yerr_data = df[yerr].to_numpy()
    else:
        yerr_data = np.zeros(len(value_data))
    if variable==parameter:
        fitter=" "
    if value==parameter:
        fitter=" "
    #print(pool_radius, route_ratio, parameter_data)
    unique_parameter = np.unique(parameter_data)
    if len(unique_parameter)==1:
        suppress_cbar=True
    # print(parameter_data,value_data, pool_radius)
    if len(unique_parameter)>maxlegendlen or cbar:
        legend = False
        cbar = True
    else:
        legend = True
    if discrete:
        evenly_spaced_interval = np.linspace(0, 1, len(unique_parameter))
        colors = [colormap(x) for x in evenly_spaced_interval]
    elif cscale=="log":
        log_spaced_interval = np.log(unique_parameter)
        log_spaced_interval = log_spaced_interval/log_spaced_interval.max()
        colors = [colormap(x) for x in log_spaced_interval]
    else:
        evenly_spaced_interval = (unique_parameter-unique_parameter.min())/(unique_parameter.max()-unique_parameter.min())
        colors = [colormap(x) for x in evenly_spaced_interval]
    # print('colors',colors)
    # for i in range(len(colors)):
        # print(colors[i], evenly_spaced_interval[i], unique_parameter[i])
    if parameter == None:
        legend = False
        if color == None:
            colors = ['green']
        else:
            colors = [color]
    if suppress_cbar:
        cbar=False
    if cbar:
        fig, (axis, cax) = plt.subplots(ncols=2, figsize=figsize,
                                  gridspec_kw={"width_ratios": [1, 0.05]})
    else:
        fig, axis = plt.subplots(ncols=1, figsize=figsize)
    if fitter != None:
        all_fitter_vals = fitter.replace(' ','').split(',')
    else:
        all_fitter_vals = []
    fitted = False
    result_df = pd.DataFrame(columns=['function',parameter,'R2'])
    last_label = None
    last_popt = None
    for counter in range(len(unique_parameter)):
        parameter_value = unique_parameter[counter]
        if len(str(parameter_value).split("."))>1 and len(str(parameter_value).split(".")[-1])>=4:
            print_parameter_value = round(float(parameter_value))
            sign = r" $\approx$ "
        elif len(parameterstr)==0:
            print_parameter_value =parameter_value
            sign = ""
        else:
            print_parameter_value =parameter_value
            sign = r" = "
        indices = np.where(parameter_data == parameter_value)[0]
        # specify if fitter where for all
        if all_fitters:
            fitter_vals=all_fitter_vals
        else:
            try:
                fitter_vals=[all_fitter_vals[counter]]
            except IndexError: # if not enough use last one
                fitter_vals = [all_fitter_vals[-1]]
        if all_fitters:
            p0=p0_in
        else:
            try:
                p0=p0_in[counter]
            except (IndexError,TypeError):
                p0=None
        if yerr != None and yerr != "off":
            unique_pool_radius, means, errs = calc_mean_with_error(np.array([pool_radius[indices], value_data[
                indices], yerr_data[indices], xerr_data[indices]]))
            value_by_pool_radius = means[0]
            value_by_pool_radius_err = means[1]
            pool_radius_err = means[2]
        else:
            # print(np.array([pool_radius[indices], value_data[indices]]))
            # print(np.array([pool_radius[indices], value_data[indices]]).dtype, pool_radius.dtype, value_data.dtype)
            unique_pool_radius, means, errs = calc_mean_with_error(np.array([pool_radius[indices], value_data[indices], xerr_data[indices]]))
            value_by_pool_radius = means[0]
            value_by_pool_radius_err = errs[0]
            pool_radius_err = means[1]
        if legend:
            if type(parameter_value)!=str:
                if parameter_value >= 2147483647:
                    this_label = parameterstr +sign+"$\infty$"
                else:
                    this_label = parameterstr +sign+str(print_parameter_value)
            else:
                this_label = parameterstr + sign + str(print_parameter_value)
        else:
            this_label = None
        # print(parameter_value, colors[counter])
        if bars:
            axis.bar(unique_pool_radius, value_by_pool_radius, color=colors[counter],alpha=0.5,width=0.8,
                      label=this_label)
        else:
            if step:
                downsteps = np.append(np.diff(unique_pool_radius)[0],np.diff(unique_pool_radius))*0.99999/2
                upsteps =np.append(np.diff(unique_pool_radius),np.diff(unique_pool_radius)[-1])*0.99999/2
                df1 = pd.DataFrame(data=np.array([unique_pool_radius-downsteps,value_by_pool_radius]).T,columns=["x","y"])
                df1["sorter"]=1
                df2 = pd.DataFrame(data=np.array([unique_pool_radius+upsteps,value_by_pool_radius]).T,columns=["x","y"])
                df2["sorter"]=2
                dfplot=df1.append(df2).sort_values(["x","sorter"])
                axis.plot(dfplot.x,dfplot.y,marker="",  linestyle="-", color=colors[counter],label = this_label)
            else:
                axis.plot(unique_pool_radius, value_by_pool_radius,
                  marker=marker, markersize=markersize, linestyle=linestyle, color=colors[counter], label = this_label)
        if yerr != "off":
            axis.errorbar(unique_pool_radius, value_by_pool_radius, yerr=value_by_pool_radius_err, xerr=pool_radius_err
                      ,marker="",alpha=0.5, capsize=5, linestyle="", color=colors[counter])
        functions = []
        R2s = []
        labels = []
        popts = []
        for fit_counter in range(len(fitter_vals)):
            if "number_of_transporters" ==parameter:
                B = parameter_value
            fitter = fitter_vals[fit_counter]
            precision = precision
            function, fitlabel, p0 = set_fit_function(fitter, lam, v, B, lmax, fitted, last_label, last_popt, p0, ts,
                                                      parameter_value)
            # print(fitlabel)
            if function != None:
                try:
                    # print('p0',p0)
                    popt, pcov = curve_fit(function, unique_pool_radius, value_by_pool_radius, p0=p0)
                    y_pred = function(unique_pool_radius, *popt)
                    R2 = r2_score(value_by_pool_radius, y_pred)
                    if True:#R2 > 0.7:
                        this_label = fitlabel
                        this_label = this_label.replace("+-", "-")  # if any inserted value is negative
                        this_label = this_label.replace("--", "+")  # if any inserted value is negative
                        if short_value_label != None:
                            this_label=this_label.replace('y=', short_value_label + "=")
                        R2s+= [R2]
                        functions += [function]
                        labels+= [this_label]
                        popts+= [popt]
                except RuntimeError:
                    print("RuntimeError for",parameter_value, 'and fitter', fitter)
        if len(R2s)>0:
            R2s = np.asarray(R2s)
            best_fit = np.argmax(R2s)
            R2 = R2s[best_fit]
            popt = popts[best_fit]
            fitlabel = labels[best_fit]
            function = functions[best_fit]
            if xlim==None:
                span = (unique_pool_radius.max()-unique_pool_radius.min())/50
                plot_pool_radius= np.arange(unique_pool_radius.min(),unique_pool_radius.max()+span,span)
            else:
                # print(xlim)
                span = (xlim[1]-xlim[0])/50
                plot_pool_radius= np.arange(xlim[0],xlim[1]+span,span)
            empty_fitlabel = fitlabel
            # no labels because they are to large, results in result_df
            if legend and len(unique_parameter)<3 or parameter == None and not abstract_fit:
               this_label = fitlabel + '\n$R^2=%5.4f' % (R2) + '$'
               for str_counter in range(len(popt)):
                   this_label =this_label.replace('{'+ascii_lowercase[str_counter]+'}',
                                                  str(round(float(popt[str_counter]), precision)))
               this_label = this_label.replace("+-","-") # if any inserted value is negative
               this_label = this_label.replace("--","+") # if any inserted value is negative
               if short_value_label!=None:
                   this_label=this_label.replace('y=',short_value_label+"=")
               if short_axis_label!=None:
                   this_label=this_label.replace('{x}',short_axis_label)
            else:
                this_label = None
            y_fit=function(plot_pool_radius, *popt)
            axis.plot(plot_pool_radius, y_fit, color=colors[counter],
                         linewidth=linewidth,linestyle='--',label=this_label)
            this_fit_results = [fitlabel,parameter_value,R2,*popt]
            popt_str = [ascii_lowercase[i] for i in range(len(popt))]
            this_result_df = pd.DataFrame(data=[this_fit_results], columns=['function', parameter, 'R2', *popt_str])
            result_df = pd.concat((result_df,this_result_df))
            fitted = True
            last_label = fitlabel
            last_popt = popt
        if fitted:
            if mark_max:
                max_index = np.argmax(y_fit)
                axis.plot(plot_pool_radius[max_index], y_fit[max_index],marker="X", markersize=markersize,color="red")
                # axis.annotate("",  # "{0}".format(round(value_by_pool_radius[max_index],2)),  # this is the text
                #               (plot_pool_radius[max_index], y_fit[max_index]),
                #               # this is the point to label
                #               textcoords="offset points",  # how to position the text
                #               xytext=(5, -30),  # distance from text to points (x,y)
                #               ha='center', size=small_font, arrowprops=dict(facecolor=colors[counter], shrink=0.05))
            if mark_min:
                min_index = np.argmin(y_fit)
                axis.plot(plot_pool_radius[min_index], y_fit[min_index],marker="X", markersize=markersize,color="red")
                # pos = (30, 30)
                # axis.annotate("",
                #               # "{0}".format(round(value_by_pool_radius[min_index], 2)),# this is the text
                #               (unique_pool_radius[min_index], value_by_pool_radius[min_index]),
                #               # this is the point to label
                #               textcoords="offset points",  # how to position the text
                #               xytext=pos,  # distance from text to points (x,y)
                #               ha='center', size=small_font, arrowprops=dict(facecolor=colors[counter], shrink=0.05))
                # min_index2 = np.argmin(value_by_pool_radius)
                # if min_index2 != min_index:
                #     axis.annotate(
                #         "{0}".format(round(value_by_pool_radius[min_index2], 2)),  # this is the text
                #         (unique_pool_radius[min_index2], value_by_pool_radius[min_index2]),
                #         # this is the point to label
                #         textcoords="offset points",  # how to position the text
                #         xytext=(0, 10),  # distance from text to points (x,y)
                #         ha='center', size=small_font)
        else:
            if mark_max:
                max_index = np.argmax(value_by_pool_radius)
                axis.annotate("",  # "{0}".format(round(value_by_pool_radius[max_index],2)),  # this is the text
                              (unique_pool_radius[max_index], value_by_pool_radius[max_index]),
                              # this is the point to label
                              textcoords="offset points",  # how to position the text
                              xytext=(5, -30),  # distance from text to points (x,y)
                              ha='center', size=small_font, arrowprops=dict(facecolor=colors[counter], shrink=0.05))
            if mark_min:
                min_index = np.argmin(value_by_pool_radius[:-1])
                pos = (30, 30)
                axis.annotate("",
                              # "{0}".format(round(value_by_pool_radius[min_index], 2)),# this is the text
                              (unique_pool_radius[min_index], value_by_pool_radius[min_index]),
                              # this is the point to label
                              textcoords="offset points",  # how to position the text
                              xytext=pos,  # distance from text to points (x,y)
                              ha='center', size=small_font, arrowprops=dict(facecolor=colors[counter], shrink=0.05))
                min_index2 = np.argmin(value_by_pool_radius)
                if min_index2 != min_index:
                    axis.annotate(
                        "{0}".format(round(value_by_pool_radius[min_index2], 2)),  # this is the text
                        (unique_pool_radius[min_index2], value_by_pool_radius[min_index2]),
                        # this is the point to label
                        textcoords="offset points",  # how to position the text
                        xytext=(0, 10),  # distance from text to points (x,y)
                        ha='center', size=small_font)
    if fitted:
        plotname = plotname[:-4] + "_fit.svg"
    if type(constant)==list:
        for constant_val in constant:
            breaksoon=False
            if type(constant_val)!=list:
                constant_val=constant # only one entry
                breaksoon=True
            constlw = linewidth
            constls = '--'
            if len(constant_val) >= 4:
                constlw = constant_val[3]
            if len(constant_val) >= 5:
                constls = constant_val[4]
            if len(constant_val) >= 3:
                axis.plot([pool_radius.min(),pool_radius.max()], constant_val[0]*np.ones(2), color=constant_val[1],
                          label=constant_val[2],zorder=0,
                          linewidth=constlw,linestyle=constls)
            if breaksoon:
                break
    elif type(constant)==int or type(constant)==float:
            axis.plot(pool_radius, constant*np.ones(len(pool_radius)), color='black', linewidth=linewidth,linestyle='--')
    if type(analytic)==list:
        if type(analytic[0])!=list:
            allanalytic = [analytic]
        else:
            allanalytic = analytic
        for analytic in allanalytic:
            alw = linewidth
            als = '--'
            if len(analytic) >= 4:
                alw = analytic[3]
            if len(analytic) >= 5:
                als = analytic[4]
            if len(analytic) >= 3:
                span = (pool_radius.max() - pool_radius.min()) / 50
                if xscale == "log":
                    plot_pool_radius = np.exp(
                        np.linspace(np.log(pool_radius.min()), np.log(pool_radius.max() + span), 50))
                else:
                    if xlim != None:
                        xmin=xlim[0]
                        xmax=xlim[1]
                    else:
                        xmin=pool_radius.min()
                        xmax=pool_radius.max()
                    plot_pool_radius = np.linspace( xmin,xmax+ span, 50)
                y_analytic = analytic[0](pool_radius)
                # print('plot_pool_radius',plot_pool_radius)
                R2_analytic = r2_score(value_data, y_analytic)
                if R2_analytic > 0.9:
                    label = analytic[2].replace('R2', '\n$R^2=%5.4f' % (R2_analytic) + '$')
                else:
                    label = analytic[2]
                axis.plot(plot_pool_radius, analytic[0](plot_pool_radius), color=analytic[1],
                          label=label, zorder=0, linewidth=alw, linestyle=als)
            else:
                print('analytic has wrong arguments')
                raise ValueError
        # %% plot arrow
        if len(arrow) > 0:
            if len(np.array(arrow).shape) == 1:  # only one arrow
                arrows = [arrow]
            else:
                arrows = arrow
            for arrow in arrows:
                if len(arrow) > 4:
                    color = arrow[4]
                else:
                    color = 'red'
                axis.annotate("", xy=arrow[2:4], xytext=arrow[:2],
                              arrowprops=dict(arrowstyle="->, head_width=0.5, head_length=0.5", color=color,
                                              linewidth=linewidth))

        if len(point) > 0:
            if len(np.array(point).shape) == 1:  # only one arrow
                points = [point]
            else:
                points = point
            for point in points:
                if len(point) > 3:
                    if type(point[3]) == list:
                        dx = point[3][0] * (unique_pool_radius.max() - unique_pool_radius.min()) / 100
                        dy = point[3][1] * (value_data.max() - value_data.min()) / 100
                    else:
                        dx = point[3] * (unique_pool_radius.max() - unique_pool_radius.min()) / 100
                        dy = point[3] * (value_data.max() - value_data.min()) / 100
                if len(point) > 4:
                    color = point[4]
                else:
                    color = "b"
                if len(point) > 5:
                    marker = point[5]
                else:
                    marker = "o"
                axis.annotate(point[2], xy=point[:2], xytext=(point[0] + dx, point[1] + dy), color=color,
                              fontsize=small_font)
                axis.plot(*point[:2], color=color, marker=marker, markersize=markersize, mew=linewidth)
            # axis.arrow(arrow[:2],arrow[2]-arrow[0],arrow[3],arrow[1], color="r")
    if title != None:
        if title == "":
            top = 0.98
        else:
            fig.suptitle(title, fontsize=large_font)
            # print('\n' in title, title)
            if '\n' in title:
                top = 0.83
            else:
                top = 0.88
    else:
        top = 0.88
        if parameterstr != None:
            fig.suptitle(valuelabel + ' vs ' + axislabel + ' by ' + parameterstr, fontsize=large_font)
        else:
            fig.suptitle(valuelabel + ' vs ' + axislabel, fontsize=large_font)
    #if short_axis_label == None:
    axis.set_xlabel(axislabel, {'fontsize': middle_font})
    #else:
    #axis.set_xlabel(short_axis_label, {'fontsize': middle_font})
    axis.set_ylabel(valuelabel, {'fontsize': middle_font})
    if variable=="pool_radius":
        print(pool_radius.max())
        axis.set_xlim((0,pool_radius.max()))#0.25
    if xlim!=None:
        axis.set_xlim(xlim)
    if ylim!=None:
        axis.set_ylim(ylim)
    else:
        if min(value_data) == 0:
            axis.set_ylim(bottom=0)
        elif min(value_data) > 0:
            axis.set_ylim(ymin=0)
    ymin,ymax=axis.get_ylim()
    # print('ymin,ymax',ymin,ymax)
    if type(vline) == list:
        constlw = linewidth
        constls = '--'
        if len(vline) >= 4:
            constlw = vline[3]
        if len(vline) >= 5:
            constls = vline[4]
        if len(vline) >= 3:
            axis.vlines(vline[0], ymin, ymax, color=vline[1], label=vline[2], linewidth=constlw, linestyle=constls,
                        zorder=0)
    elif type(vline) == int or type(constant) == float:
        print(vline, ymin, ymax)
        axis.vlines(vline, ymin, ymax, color='black', linewidth=linewidth, linestyle='--')
    axis.tick_params(labelsize=small_font, width=thin_linewidth, length=linewidth)
    if legendpos == None:
        legendpos ="best"
    if cbar: #no legend but colorbar
        if cscale=="log":
            normalize = mcolors.LogNorm(vmin=unique_parameter[unique_parameter>0].min(), vmax=unique_parameter[
                    unique_parameter>0].max())
        elif cscale=="linear":
            normalize = mcolors.Normalize(vmin=unique_parameter[unique_parameter>=0].min(), vmax=unique_parameter[
                    unique_parameter>=0].max())
        # print('normalize',normalize)
        else:
            print("no cbar defined for this scale")
        cbar = add_discrete_cbar(normalize, colormap, unique_parameter, fig, small_font, middle_font, thin_linewidth,
                                 linewidth, parameterstr, axis = cax, precision=precision)
        handles, labels = axis.get_legend_handles_labels()
        if fitted:
            handles += [Line2D([pool_radius[0]], [value_data[0]], color="grey", linewidth=3, linestyle='--')]
            if short_axis_label != None:
                empty_fitlabel = empty_fitlabel.replace('{x}',short_axis_label)
            if parameterset == 'greek':
                print('before',empty_fitlabel)
                empty_fitlabel = empty_fitlabel.replace('{a}',r'\alpha')
                empty_fitlabel = empty_fitlabel.replace('{b}',r'\beta')
                empty_fitlabel = empty_fitlabel.replace('{c}',r'\gamma')
            labels += [empty_fitlabel]
        if len(handles)>0 and not suppress_legend:
            axis.legend(handles, labels,fontsize=small_font, loc=legendpos)
    else:
        handles, labels = axis.get_legend_handles_labels()
        if fitted and (len(unique_parameter)>=3 or abstract_fit):
            handles += [Line2D([pool_radius[0]], [value_data[0]], color="grey", linewidth=3, linestyle='--')]
            if short_axis_label != None:
                empty_fitlabel = empty_fitlabel.replace('{x}', short_axis_label)
                empty_fitlabel = empty_fitlabel.replace('^{-1/', '^{')
            if parameterset == 'greek':
                # print('before',empty_fitlabel)
                empty_fitlabel = empty_fitlabel.replace('{a}',r'\alpha')
                empty_fitlabel = empty_fitlabel.replace('{b}',r'\beta')
                empty_fitlabel = empty_fitlabel.replace('{c}',r'\gamma')
                # print('after',empty_fitlabel)
            labels += [empty_fitlabel]
        if len(handles) > 0 and not suppress_legend:
            axis.legend(handles, labels, fontsize=small_font, loc=legendpos)
    # unify legendlabels
    if sum_legend:
        handles=[]
        labels=[]
        cur_handles, cur_labels=axis.get_legend_handles_labels()
        # print(cur_labels)
        if len(cur_labels)>=2*len(unique_parameter):
            parametervalinlegend=2
        else:
            parametervalinlegend=1
        for counter in range(len(unique_parameter)):
            parameter_value = unique_parameter[counter]
            handles += [Line2D([pool_radius[0]], [value_data[0]], color=colors[counter], linewidth=linewidth,
                               linestyle='--', marker="o", markersize=markersize)]
            labels += [cur_labels[counter*parametervalinlegend+parametervalinlegend-1].replace('y',parameter_value)]
        if len(handles) > 0:
            axis.legend(handles, labels, fontsize=small_font, loc=legendpos)
    # print(yscale)
    axis.set_yscale(yscale)
    axis.set_xscale(xscale)
    if 'x' in scientific_axis:
        print('scientific x')
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(ScalarFormatter())
        plt.setp(axis.get_xticklabels(minor=False), visible=True)
        plt.setp(axis.get_xticklabels(minor=True), visible=False)
        axis.ticklabel_format(axis='x', style='sci', scilimits=(3, 4))
    else:
        axis.ticklabel_format(axis='x', style='plain')
    if 'y' in scientific_axis:
        axis.yaxis.set_major_formatter(ScalarFormatter())
        axis.ticklabel_format(axis='y', style='sci', scilimits=(3, 4))
    else:
        axis.ticklabel_format(axis='y', style='plain')
    if cbar:
        if 'c' in scientific_axis:
            cax.ticklabel_format(axis='x',style='sci', scilimits=(3, 4))
    #%% add second axis simply with different scale
    if len(secondaxis)>0:
        secax = axis.secondary_yaxis('right', functions=(secondaxis[1], secondaxis[2]))
        secax.set_ylabel(secondaxis[0])
    # print(top)
    plt.tight_layout()
    fig.subplots_adjust(top=top)
    # print('top',top, title)

    if parameterstr != None:
        this_plot_suffix = value+'_vs_'+variable+"_"+parameter+"_"
    else:
        this_plot_suffix = value+'_vs_'+variable+"_"
    print(directory + this_plot_suffix + plotname)
    if os.path.isfile(directory + this_plot_suffix + plotname):
        os.remove(directory + this_plot_suffix + plotname)
    plt.savefig(directory + this_plot_suffix + plotname)
    plt.show()
    try:
        if parameterstr != None:
            sf.store_plotdata(directory, this_plot_suffix + plotname[:-3] + "csv",
                              np.array([pool_radius, value_by_pool_radius, parameter_data])
                              , variable + ', ' + value + ', ' + parameter)
        else:
            sf.store_plotdata(directory, this_plot_suffix + plotname[:-3] + "csv",
                          np.array([pool_radius, value_by_pool_radius])
                          , variable+', '+value)
    except TypeError: #one column contains strings, not floats
        df[keep_columns].to_csv(directory+this_plot_suffix+plotname[:-4]+".csv")
    #print(fit_results)
    if len(result_df)>0:
        return result_df
