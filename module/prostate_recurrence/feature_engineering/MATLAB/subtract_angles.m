function [angle_difference] = subtract_angles(x,Y)
%SUBTRACT_ANGLES produces a vector of absolute differences in angle between
% a scalar, x and a vector, Y where x,Y have a range of pi radians.
%
%   [angle_difference] = subtract_angles(x,Y) accounts for periodicity in 
%   angle subtraction such that angular difference falls in the range
%   [0,pi].
%
%   Input:
%   x: Scalar angle in radians
%   Y: Vector of radian values.
%
%   Output:
%   angle_difference: a vector of differences in angle between x and all
%   values in Y, in range, [0,pi].
%
%
%   Class support for input x, Y:
%      float: single, double
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.
angle_region1 = (abs(x-Y)>pi/2).*(pi-abs(x-Y));
angle_region2 = (abs(x-Y)<=pi/2).*abs(x-Y);
angle_difference = max(angle_region1,angle_region2);

end