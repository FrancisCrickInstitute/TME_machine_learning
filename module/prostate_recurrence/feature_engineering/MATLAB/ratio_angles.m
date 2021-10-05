function [angle_ratio] = ratio_angles(x,Y)
%RATIO_ANGLES produces a vector of ratios between a scalar, x and a vector,
%   Y. Instance of Y=0 are removed.
%
%   [angle_ratio] = ratio_angles(x,Y) calculates the ratio between a
%   scalar, x, and vector, Y for all non-zero elements of Y.
%
%   Input:
%   x: Scalar.
%   Y: Vector.
%
%   Output:
%   angle_ratio: a vector of ratios of x to all non-zero values in Y.
%
%
%   Class support for input x, Y:
%      float: single, double
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.
Y=Y(Y>0);
angle_ratio = x./Y;

end