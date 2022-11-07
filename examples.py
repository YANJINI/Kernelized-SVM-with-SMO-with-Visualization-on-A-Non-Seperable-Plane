import sys
sys.path.extend(['/Users/jinijani/PycharmProjects/git_project/ML/Kernelized_SVM_with_SMO'])

from kernelized_svm import SMO_kernel_SVM

a = SMO_kernel_SVM(kernel='linear', C=1, max_iter=60, degree=3, gamma=1)
a = SMO_kernel_SVM(kernel='rbf', C=10, max_iter=60, degree=3, gamma=1)