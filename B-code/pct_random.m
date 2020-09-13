function [ RXS, RXL ] = pct_random( XS,XL,npercent )
% INPUT: XS,XL are 'n*d','n*1'
% OUTPUT: RXS,RXL are 'n1*d','n1*1'( n1 = n * npecent )
nClass = length(unique(XL));  % The number of classes;
num_Class=[];
for i=1:nClass
  num_Class = [num_Class length(find( XL == i ))]; %The number of samples of each class
end
RXS=[];
RXL=[];

for  j=1:nClass
    sele_num = ceil( npercent * num_Class(j) ) ;  %  select training samples  
    idx = find( XL == j );
    randIdx=randperm(num_Class(j));
    RXS = [RXS; XS(idx(randIdx(1:sele_num)),:)]; % Random select select_num samples per class for training
    RXL = [RXL; XL(idx(randIdx(1:sele_num)))];
end
