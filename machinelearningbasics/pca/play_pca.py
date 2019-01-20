import numpy as np;

np.random.seed(123123124);
mu_vec1 = np.array([0,0,0]);
#print(type(mu_vec1));
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]]);
#print(type(cov_mat1));

class1_sample = np.random.multivariate_normal(mean = mu_vec1, cov = cov_mat1, size = 20).T
#print(class1_sample);
#print(type(class1_sample));

mu_vec2 = np.array([1,1,1]);
class2_sample = np.random.multivariate_normal(mean = mu_vec2, cov = cov_mat1, size = 20).T
#print(class2_sample);

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D;
from mpl_toolkits.mplot3d import proj3d;
fig = plt.figure(figsize=(8,8));
ax = fig.add_subplot(111, projection = '3d');
plt.rcParams['legend.fontsize'] = 10
ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color = 'blue', alpha = 0.5, label = 'class1');
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, color = 'red', alpha = 0.5, label = "class2");

plt.title('Samples for class 1 and class 2');
ax.legend(loc='upper right');
#plt.show();

print("class1 sample");
print(class1_sample);
print("class2 sample");
print(class2_sample);

all_samples = np.concatenate((class1_sample, class2_sample), axis=1);
print(all_samples);
print(all_samples.shape);
mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

print('Mean Vector:\n', mean_vector)

scatter_matrix = np.zeros((3,3))
for i in range(all_samples.shape[1]):
    scatter_matrix += (all_samples[:,i].reshape(3,1) - mean_vector).dot((all_samples[:,i].reshape(3,1) - mean_vector).T)
print('Scatter Matrix:\n', scatter_matrix)


cov_mat = np.cov([all_samples[0, :], all_samples[1, :], all_samples[2, :]]);
print("covariance matrix");
print(cov_mat);
print('shape: ',all_samples.shape);
cov_mat = cov_mat * (all_samples.shape[1] - 1);
print("cov_matr * length");
print(cov_mat);

eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat);
print('eigen vector: ');
print(eig_vec_cov[0, :]);
eig_vc = np.reshape(eig_vec_cov[:,0], newshape=(3,1));
print('eigen vector');
print(eig_vc);

print('covariance matrix dot eigenvector: \n');
print(cov_mat.dot(eig_vc));
print('eigen value multiplies eigenvector: \n');
print(eig_val_cov[0]*eig_vc);

eig_pairs = [ (np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))];
print('eig_pairs \n',eig_pairs);
eig_pairs.sort(key=lambda x: x[0], reverse = True);
for i in eig_pairs:
    print(i[0]);

matrix_w = np.hstack((eig_pairs[0][1].reshape((3,1)), eig_pairs[1][1].reshape(3,1)) )
print(matrix_w);
transformed = matrix_w.T.dot(all_samples);
print(transformed);
print(transformed.shape);
