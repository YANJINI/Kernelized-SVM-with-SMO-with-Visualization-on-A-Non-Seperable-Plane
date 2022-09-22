# Kernelized_SVM_with_SMO
Application of kernelized SVM with SMO algorithm to non-linearly seperable 2D case

## Bacic Idea
From an inspiration by [kernel SVM demo](https://youtu.be/RwF1esLCG4U?t=2801) on lecture 24 of prof. Kilian Weinberger's [Machine Learning for Intelligent Systems course](https://www.cs.cornell.edu/courses/cs4780/2018fa/) at Cornell University, I have built conceptually same interactive kernel SVM demo in Python with matplotlib event handling. For solving SVM optimization problem, I have used Sequential Minimal Optimization (SMO) algorithm in reference to [this paper regarding SMO algorthm to solve SVM without any libraries (https://www.researchgate.net/publication/344460740_Yet_more_simple_SMO_algorithm). <br />


## How it works
### Linear Kernel SVM

on the figure that pops up, you click to plot from class 1 and 2 as follows. <br />

![click to plot](/images/click_to_plot_linear.gif)


Pressing enters makes SMO argorithm find the optimized lamdas (Lagrange multiplier) and show the result. <br />
![linear kernel](/images/linear_kernel.gif)

### RBF Kernel

plot non-linear data points the same way as above.
![click to plot](/images/click_to_plot_circle.gif)

Pressing enters makes SMO argorithm find the optimized lamdas (Lagrange multiplier) and show the result. <br />
![rbf kernel](/images/rbf_kernel.gif)
<br />
<br />

## Setup

### git clone
git clone to have this repository on your local machine as follows.
```ruby
git clone git@github.com:YANJINI/Kernelized_SVM_with_SMO.git
```

### path control
To import the classifier modules as done on examples.py, you have to control path to these modules as below (Mac OS)
```ruby
import sys
sys.path.extend(['/path_to_this_repository/Kernelized_SVM_with_SMO.git'])
```

### import 
Import these classifiers to another py project as below.
```ruby
from kernelized_svm import SMO_kernel_SVM
```

### Others
Check examples.py to see how to use the programes
