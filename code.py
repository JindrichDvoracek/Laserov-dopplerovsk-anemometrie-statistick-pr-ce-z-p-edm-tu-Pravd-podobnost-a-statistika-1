# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 19:25:42 2025

@author: Jindrich Dvoracek
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import t
from scipy.stats import skew
from scipy.stats import norm

#closes all previous plots
plt.close("all")

def t_confidence_interval(data, alpha=0.05):
    #computing basic statistics and the (1-alpha)-CI
    average = np.mean(data)
    stdeva = np.std(data) #standard deviation
    se = stdeva/np.sqrt(len(data)) #standard error
    df = len(data)-1 #degrees of freedom to be passed to t-test
    t_alpha_half = t.ppf(1-(alpha/2), df)
    #returns mean, stdeva, se and lower and upper bound of 1-alpha CI
    return (average, stdeva, se,
            average + t_alpha_half*se, average - t_alpha_half*se)
        


# Loading data
data_raw = np.genfromtxt("data.csv", delimiter=";", skip_header=1)
data = data_raw[:, 5]
#splitting data into two batches for statistic testing
batch1 = data[::2]
batch2 = data[1::2]

###Explorative analysis###
#Optimal no. of bins acc. to H. A. Sturges, J.
#American Statistical Association, 65–66 (1926). 
m_opt = (np.log(len(data))/np.log(2))+1

#Plotting histograms
fig, ax = plt.subplots(nrows = 2, ncols=1)
counts1, bins1, _ = ax[0].hist(data, bins=int(np.ceil(m_opt)), edgecolor="black")
ax[0].set_xlabel(r'$v_x \: / \: mm \cdot s^{-1}$')
ax[0].set_ylabel("Counts")
# ax[0].set_title("Histogram of velocities")
#My preferred no. of bins
counts2, bins2, _ = ax[1].hist(data, bins=15, edgecolor="black")
ax[1].set_xlabel(r'$v_x \: / \: mm \cdot s^{-1}$')
ax[1].set_ylabel("Counts")

# Fit normal distribution
mu_fit, std_fit = norm.fit(data)

# Generate x values
x = np.linspace(min(data), max(data), 1000)
pdf = norm.pdf(x, mu_fit, std_fit)

# Scale PDF for first histogram 
bin_width1 = bins1[1] - bins1[0]
pdf_scaled1 = pdf * len(data) * bin_width1
ax[0].plot(x, pdf_scaled1, color='red', linewidth=2,
           label=f"Normal fit\n $\mu = {mu_fit:.2f}, \: \sigma = {std_fit:.2f}$")
ax[0].legend()

# Scale PDF for second histogram
bin_width2 = bins2[1] - bins2[0]
pdf_scaled2 = pdf * len(data) * bin_width2
ax[1].plot(x, pdf_scaled2, color='red', linewidth=2)

fig.tight_layout()
fig.savefig("C:/Documents/MFF UK/4. semestr LS/Pravděpodobnost a statistika 1/statisticka prace/figures/histogram.pdf", format="pdf")

#Ploting both batches
fig2, ax2 = plt.subplots(1,1)
ax2.hist(batch1, bins=15, color='blue', edgecolor="black", alpha = 0.5, label='First batch')
ax2.hist(batch2, bins=15, color='red', edgecolor="black", alpha = 0.5, label='Second batch')
ax2.set_xlabel(r'$v_x \: / \: mm \cdot s^{-1}$')
ax2.set_ylabel("Counts")
# ax2.set_title('Both batches')
ax2.legend()
fig2.tight_layout()
fig2.savefig("C:/Documents/MFF UK/4. semestr LS/Pravděpodobnost a statistika 1/statisticka prace/figures/histogram_batches.pdf", format="pdf")



###Test of normality - Shapiro–Wilk test###
stat, p = shapiro(data)
print(f"Shapiro-Wilk test of all data: W={stat:.4f}, p={p:.4f}")

stat_batch1, p_batch1 = shapiro(batch1)
print(f"Shapiro-Wilk test of first batch: W={stat_batch1:.4f}, p={p_batch1:.4f}")

stat_batch2, p_batch2 = shapiro(batch2)
print(f"Shapiro-Wilk test of second batch: W={stat_batch2:.4f}, p={p_batch2:.4f}")

###Interval estimate of mu and sigma###

#computing mu, sigma and rejection region from the first batch
mu, sigma, se, upper, lower = t_confidence_interval(batch1)
print("First batch's statistics below:")
print(f'Mu = {mu:.2f}, sigma = {sigma:.2f}, se = {se:.2f}, the 95% confidence interval for mu is: [{lower:.2f}, {upper:.2f}]')

#performing estimate of mu from the second batch
average2 = np.mean(batch2)
print(f"Mean for the second batch is: {average2:.2f}")
if average2 <=upper and average2>=lower:
    print(f"Mu estimate mu = {average2:.2f} lays in the 95% confidence interval.")
else:
    print(f"Mu estimate {average2:.2f} does not lay in the 95% confidence interval.")

#Ploting both batches with vertical lines
fig3, ax3 = plt.subplots(1,1)
ax3.hist(batch1, bins=15, color='blue', edgecolor="black",
         alpha = 0.5, label='First batch')
ax3.hist(batch2, bins=15, color='red', edgecolor="black",
         alpha = 0.5, label='Second batch')
ax3.set_xlabel(r'$v_x \: / \: mm \cdot s^{-1}$')
ax3.set_ylabel("Counts")
ax3.vlines((mu, lower, upper), 0, 10, color='blue')
ax3.vlines(average2, 0, 10, color='red')
ax3.legend()
fig3.tight_layout()
fig3.savefig("C:/Documents/MFF UK/4. semestr LS/Pravděpodobnost a statistika 1/statisticka prace/figures/histogram_batches_lines.pdf", format="pdf")


###computing the 3rd moment###
skew = skew(data)
print(f"Skewness is: {skew:.4f}.")

plt.show()