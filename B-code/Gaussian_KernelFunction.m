function [G] = Gaussian_KernelFunction(pattern1,pattern2,deg)

size1 = size(pattern1);
size2 = size(pattern2);

R1 = sum((pattern1.*pattern1),2);
R2 = sum((pattern2.*pattern2),2);

M1 = repmat(R1,1,size2(1));
M2 = repmat(R2',size1(1),1);
M3 = 2 * pattern1 * pattern2';
G = M1 - M3 + M2; 
G = exp(-G/2/deg^2);