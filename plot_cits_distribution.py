#! /usr/bin/env python
#
# usage: ./plot_cits_distribution.py cits.txt

import sys
import powerlaw
import matplotlib.pylab as plt
import numpy as np

def top_papers(data, best=5):

    X = []
    Y = []
    for data_point in data:
       X.append(data_point[0])
       Y.append(data_point[1])

    XY = zip(Y,X)
    XY.sort()
    return XY[-(best+1):-1]



def find_pdf_at_x(x,bins,f):
    
    for i in range( len(f) ):
       if x > bins[i] and bins[i+1]:
           x_c = 0.5 * (bins[i+1] + bins[i] )
           y_c = f[i]

    return (x_c, y_c)


def fitted_pl_xy(alpha, xc, yc, xmin, xmax):
    x_ = np.linspace(xc, xmax, num=50)
    y_ = x_**(-alpha)
    scale = yc / y_[0]
    y_ *= scale

    x_ = np.linspace(0.1*xc, 10*xmax, num=50)
    y_ = x_**(-alpha)
    y_ *= scale

    return x_, y_

def create_cits_plots(data, title_=""):
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7) )
    fig, ax1 = plt.subplots(1, 1, figsize=(7,7) )
    fit = powerlaw.Fit(data, discrete=True)
    plt.suptitle(title_,fontsize=30)

    a = fit.power_law.alpha
    xmin = fit.power_law.xmin

    pdf = powerlaw.pdf(data)
    bins = pdf[0]
    widths = bins[1:] - bins[0:-1]
    centers = bins[0:-1] + 0.5*widths

    xc,yc = find_pdf_at_x(xmin, pdf[0], pdf[1])
    x_, y_ = fitted_pl_xy(a, xc, yc, xmin, np.max(pdf[0]) )

#### AX1
    ax1.set_xlabel("Number of citations", fontsize=20)
    ax1.set_ylabel("Probability density function", fontsize=20)
    ax1.plot(centers,pdf[1],'o')
    powerlaw.plot_pdf(data, ax=ax1, color='b',label='APS data')
    ax1.plot(x_, y_, 'r--', label=r'Power law fit: $x^{-%4.3f}$' % ( a ) )
#    ax1.plot(xc,yc,'o',color='grey')
    ax1.plot([xmin,xmin],[10**(-16),1],'--',color="grey", label=r'$\rm x_{min}=%d$' % (xmin))
    ax1.legend(loc=0, fontsize=15)

#### AX2
#    ax2.set_xlabel(r'$\rm x_{min}$', fontsize=20)
#    ax2.set_ylabel(r'$\rm D, \sigma, \alpha$', fontsize=20)
#    ax2.plot(fit.xmins, fit.Ds, label=r'$D$')
#    ax2.plot(fit.xmins, fit.sigmas, label=r'$\sigma$', linestyle='--')
#    ax2.plot(fit.xmins, fit.sigmas/fit.alphas, label=r'$\sigma /\alpha$', linestyle='--')
#    ax2.legend(loc=0, fontsize=15)
#    ax2.set_xlim( [0, 200] )
#    ax2.set_ylim( [0, .25] )

    fig.text(0.95, 0.05, '(c) 2018, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    plt.show()

if __name__ == "__main__":

    data = []
    full_data = []
    fin = open('../db_text/cits.txt','rU')
    for line in fin:
        pairs = line.split()
        data.append( int(pairs[1]) )
        full_data.append([pairs[0],int(pairs[1])])

    print top_papers(full_data)

    create_cits_plots(data, "All APS papers")
    

    fit = powerlaw.Fit(data, discrete=True)
    other_dists = ['exponential', 'truncated_power_law', 'lognormal', 'stretched_exponential']
    for dist_ in other_dists:
        R, p = fit.distribution_compare('power_law', dist_, normalized_ratio=True)
        print(dist_, R, p)
    
    print "Alpha: ",  fit.truncated_power_law.parameter1
    print "Lambda: ", fit.truncated_power_law.parameter2
