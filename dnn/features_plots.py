#! /usr/bin/env python

import sys
import numpy as np
import powerlaw
import matplotlib.pylab as plt
from collections import Counter

from nnutils import read_sentences


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
    ax1.set_xlabel("Number of word occurances", fontsize=20)
    ax1.set_ylabel("Probability density function", fontsize=20)
    ax1.plot(centers,pdf[1],'o')
    powerlaw.plot_pdf(data, ax=ax1, color='b',label='APS data')
    ax1.plot(x_, y_, 'r--', label=r'Power law fit: $x^{-%4.3f}$' % ( a ) )
    #ax1.plot([xmin,xmin],[10**(-16),1],'--',color="grey", label=r'$\rm x_{min}=%d$' % (xmin))
    ax1.legend(loc=0, fontsize=15)

    fig.text(0.95, 0.05, '(c) 2018, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    plt.show()


def read_file(fname):
    fin = open(fname,'rU')
    words_num = []
    for line in fin:
        pairs = line.split()
        words_num.append( int(pairs[3]) )

    fin.close()
    return np.array(words_num)



if __name__ == "__main__":

    words_tits = read_file('./results/titles.dat')
    words_abst = read_file('./results/abstracts.dat')

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,7))
    n, bins, patches = ax1.hist(words_tits, bins=range(0,25), normed=1, facecolor='g', alpha=0.75)
    ax1.set_xlabel("# Words",fontsize=20)
    ax1.set_ylabel("Frequency",fontsize=20)
    ax1.set_title("Titles histogram",fontsize=35)
    n, bins, patches = ax2.hist(words_abst, bins=range(10,300), normed=1, facecolor='g', alpha=0.75)
    ax2.set_xlabel("# Words",fontsize=20)
    ax2.set_ylabel("Frequency",fontsize=20)
    ax2.set_title("Abstracts histogram",fontsize=30)
    f.text(0.95, 0.05, '(c) 2018, P.G.',fontsize=10, color='gray', ha='right', va='bottom', alpha=0.5)
    plt.show()
   

    SENTENCES_FILE='./data/tokenized_abstracts.dat'
    sentences = read_sentences( SENTENCES_FILE )


    all_words = []
    for sentence in sentences:
        all_words += sentence
    counts = Counter(all_words)
    x = counts.values()

    create_cits_plots(x, "All abstracts")

