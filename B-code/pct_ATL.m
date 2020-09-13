function [ RXS, RYS ] = pct_ATL( XS,XL,A,npct,d,num )
RXS = [ A  XS' ];
RXS = sortrows( RXS ,1 )';
RXS = RXS(2 :(d + 1) , npct+1 : num );
RYS = [ A XL ];
RYS = sortrows( RYS ,1 )';
RYS = RYS( 2:2 ,  npct+1 : num );
