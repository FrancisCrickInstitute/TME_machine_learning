function [x_vector,y_vector] = vector_normalisation(x_vector,y_vector)
%VECTOR_NORMALISATION Normalises vector components to unit vector.
%
%   [x_vector,y_vector] = vector_normalisation(x_vector,y_vector)
%   normalises vectors assuming x and y components are stored in different
%   matrices. For instances of zero vector magntitude it is assumed that
%   both x and y components are very small, equal and positive so that in
%   the limit, both x and y components equal 1/sqrt(2). Vectors are
%   standardised from point of view of x vector i.e. all vectors are set
%   such that x direction is always positive.
%
%   Input:
%   x_vector: Matrix of x vector values
%   y_vector: Matrix of y vector values.
%
%   Output:
%   x_vector: Normalised matrix of x vector values
%   y_vector: Normalised matrix of y vector values.
%
%
%   Class support for input x_vector, y_vector:
%      float: single, double
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.

%Vectors standardised relative to a positive x direction
y_vector(x_vector<0)=-1*y_vector(x_vector<0);
x_vector(x_vector<0)=-1*x_vector(x_vector<0);
vector_magnitude=(x_vector.^2+y_vector.^2).^0.5;
x_vector=x_vector./vector_magnitude;
y_vector=y_vector./vector_magnitude;
x_vector(vector_magnitude == 0) = 1/(2^0.5);
y_vector(vector_magnitude == 0) = 1/(2^0.5);

end

