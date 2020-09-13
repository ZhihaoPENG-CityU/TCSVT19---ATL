%% Reference
% [1] Peng, et al. "Active Transfer Learning.". in TCSVT'19.
function [ A, P, obj ] = RMMD(X,Y,KXY,WXY,Part)
[d, n1] = size(X);
[~, n2] = size(Y);
r =  floor((3*d)/4) ;
B1 = ones(n1,1);
B2 = ones(n2,1);
I = eye(n1,n1);
P = eye( d, r );

r1 = Part.r1;
r2 = Part.r2;

A = zeros( n1, 1) ;
V = zeros( n1 ,1 );
Y2 = zeros( n1,1 );
y1 = 0;
u = Part.u ;
umax = 10^5;
pu = 1.01;
MaxIter = 350;
opts.record = 0;
opts.mxitr = 150;
opts.xtol = 1e-2;
opts.gtol = 1e-2;
opts.ftol = 1e-3;
    
for i = 1:MaxIter
    % Update weight vector by Eq. (13) in [1]
    left = ( 2 / (n1^2 ) ) * ((( X' * P) * P') * X ) + r1 * (KXY + KXY') + r2 * (WXY + WXY') + u * ( B1 * B1' ) + u * I ;
    right = ( 2/(n1 * n2)) * ((( (X' * P) * P') * Y ) * B2) + u * B1 + u * V - y1 * B1 - Y2 ;
    A  = pinv(left) * right ;
    
    % Update V by Eq. (19) in [1]
    V = A + ( 1 / u ) * Y2 ;
    V = max( V ,0 );
    
    % Update projection matrix P by Eq. (16) in [1]
    [ P ,~ ] = OptStiefelGBB( P, @fun, opts, X, A, Y, B2, n1, n2 );
    
    % Update other by Eq. (20) in [1]
    y1 = y1 + u * ( A' * B1 - 1 );
    Y2 = Y2 + u * ( A - V );
    u = min(  pu * u ,umax );
    
    % disp the converge of result
    disp(['i:',num2str(i)]);
    con = norm(A-V, 'fro')^2;
    obj(i) = con;
	% repeat above updating steps until convergence
    if norm(A-V, 'fro')^2<10^-11 || norm(A-V, 'fro')^2 > 10^18 
        break;
    end
end
end