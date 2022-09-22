import sys
sys.path.extend(['/Users/jinijani/PycharmProjects/git_project/ML/Kernelized_SoftMargin_SVM_with_Visualization_on_A_Plane'])

from kernelized_svm import SMO_kernel_SVM

a = SMO_kernel_SVM(kernel='rbf', C=10, max_iter=60, degree=3, gamma=1)