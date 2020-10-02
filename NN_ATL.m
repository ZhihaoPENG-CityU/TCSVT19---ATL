%% We appreciate it if you use this matlab code and cite our papers.
%% Contact: zhihapeng3-c@my.cityu.edu.hk; zhpengcn@126.com;
%% The BibTeX files are as follows,
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
  note={doi:{\color{blue}
\href{https://doi.org/10.1109/LSP.2020.3025061}{10.1109/LSP.2020.3025061}}
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

pct1 = 0.8; 
% pct2 = 0.4;
% pct3 = 0.6;
% pct4 = 0.8;
[d,num] = size(XS_S);

% random pct + KNN classifer
[ KNN_XS_S,KNN_XS_L ] =  pct_random(XS_S',XS_L, pct1 );
acc_orignal =  KNN(KNN_XS_S,KNN_XS_L,XT_S',XT_L,1) * 100;

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

% The parameters that weight the importance of K and W respectively. (see Eq. (8) in [1])
R1 = [ 0.1 ];
R2 = [ 10000 ];  % [ 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000 ];

acc1 = [];
best1Acc = 0 ;
best1a = [] ;
npct1 = ceil( (1-pct1) * num );

for i = 1:length(R1)
    for j = 1:length(R2)
        Part.r1 = R1(i);
        Part.r2 = R2(j);
        Part.u = 0.1;
        disp(['r1:',num2str(R1(i)),'    r2:',num2str(R2(j))]);
        % Iteration Updating via the Algorithm 1 in [1]
        [ A, P, obj] = RMMD( XS_S,XT_S,K,W,Part );
        
        % Experimental strategy (see Section ¢õ in [1])
        [ XS1, YS1 ] = pct_ATL( XS_S,XS_L,A,npct1,d,num );

        acc1( (i-1)*length(R2)+j ) = KNN((P'*XS1)',YS1',(P'*XT_S)',XT_L,1) * 100;

        disp(['acc1: ',num2str(acc1)]);

        if acc1( (i-1)*length(R2)+j ) > best1Acc
            best1Acc = acc1((i-1)*length(R2)+j);
            best1a = A;
            best1r1 = R1(i);
            best1r2 = R2(j);
        end
    end
end
save('acc1.mat','acc1');
toc;
