%% We appreciate it if you use this matlab code and cite our papers.
% The BibTeX files are as follows,
%{
1- TCSVT19 --->
@article{peng2019active,
  title={Active Transfer Learning},
  author={Peng, Zhihao and Zhang, Wei and Han, Na and Fang, Xiaozhao and Kang, Peipei and Teng, Luyao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={30},
  number={4},
  pages={1022--1036},
  year={2019},
  publisher={IEEE}
}

2- SPL20  --->
@ARTICLE{9210817,
  author={Z. {Peng} and Y. {Jia} and J. {Hou}},
  journal={IEEE Signal Processing Letters}, 
  title={Non-Negative Transfer Learning with Consistent Inter-domain Distribution}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  note={doi:{\color{blue}\href{https://doi.org/10.1109/LSP.2020.3025061}{10.1109/LSP.2020.3025061}}
  }
%}
%% Reference
% [1] Peng, et al. "Active Transfer Learning.". in TCSVT'19.
%% Matlab implementation for our TCSVT'19 paper.
clc,clear;
close all;
clear memory;
currentFolder = pwd;
addpath(genpath(currentFolder));
tic;

load 'C2s vs C2t.mat'
XS_S = X_src;
XS_L = Y_src;
XT_S = X_tar;
XT_L = Y_tar;

pct = 0.8; 

[d,num] = size(XS_S);

R1 = 0.01 ; % [ 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000 ]
R2 = 0.01 ;

% Class Diversity Matrix K (see `Section ¢ò.C. The Model of ATL' in [1])
Koptions = [];
Koptions.NeighborMode = 'Supervised';
Koptions.WeightMode = 'Binary';
Koptions.gnd = XS_L;
K = constructW( XS_S',Koptions );

% Distance Restriction Matrix W (see Eq. (6) in [1])
Woptions = [];
Woptions.NeighborMode = 'KNN';
Woptions.WeightMode = 'Binary';
W = constructW( XS_S',Woptions );

acc = [] ;
bestAcc = 0 ;
besta = [] ;
npct = ceil( (1-pct) * num );

for i = 1:length(R1)
    for j = 1:length(R2)
        Part.r1 = R1(i);
        Part.r2 = R2(j);
        Part.u = 0.1;
        disp(['r1:',num2str(R1(i)),'    r2:',num2str(R2(j))]);
        [ A, P, obj] = RMMD( XS_S,XT_S,K,W,Part );
        [ XS, YS ] = pct_ATL( XS_S,XS_L,A,npct,d,num );

        SVM_XS_S = P'*XS ;
        SVM_XS_S = SVM_XS_S./repmat(sqrt(sum(SVM_XS_S.^2)),[size(SVM_XS_S,1) 1]);
        SVM_XT_S = P'*XT_S;
        SVM_XT_S  = SVM_XT_S ./repmat(sqrt(sum(SVM_XT_S.^2)),[size(SVM_XT_S,1) 1]);
        tmd = ['-s 0 -t 2 -g ' num2str(1e-3) ' -c ' num2str(1000)];
        model = svmtrain(YS', SVM_XS_S', tmd);
        [~, acc_svm] = svmpredict(XT_L, SVM_XT_S', model);
        acc( (i-1)*length(R2)+j ) = acc_svm(1);
        
        disp(['acc: ',num2str(acc)]);
        if acc( (i-1)*length(R2)+j ) > bestAcc
            bestAcc = acc((i-1)*length(R2)+j);
            besta = A;
            bestr1 = R1(i);
            bestr2 = R2(j);
        end
    end
end
toc;
