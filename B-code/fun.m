function [ F , G ] = fun( P, XS, A, XT, B, ns, nt)
G1 = ( 2/ns^2 ) * ( (((XS * A) * A') * XS') * P ) ;
G2 = ( 1/(ns * nt) ) * ( ((( XS * A) * B') * XT') * P + (((XT * B )* A') * XS') * P );
G3 = ( 1/(ns * nt) ) * ( XT * B * A' * XS' * P + XS * A * B' * XT' * P );
G4 = ( 2/nt^2 ) * ( (((XT * B) * B') * XT') * P );
F1 = ( 1/ns ) * P' * XS * A - (1/nt) * P' * XT * B;
F = norm( F1 , 'fro' );
G =  G1 - G2 -G3 + G4 ;
end