function [end_to_end_angle_continuum,end_to_end_x_derivative_continuum,end_to_end_y_derivative_continuum,local_angle_continuum,local_x_derivative_continuum,local_y_derivative_continuum] = fibre_angle_continuum(end_to_end_x_derivative_matrix,end_to_end_y_derivative_matrix,local_x_derivative_matrix,local_y_derivative_matrix)
%FIBRE_ANGLE_CONTINUUM Interpolate over NaN values to create a continuum of
%fibre angles from discrete fibre points.
%
%   [end_to_end_angle_continuum,end_to_end_x_derivative_continuum,
%   end_to_end_y_derivative_continuum,local_angle_continuum,
%   local_x_derivative_continuum,local_y_derivative_continuum] = 
%   fibre_angle_continuum(end_to_end_x_derivative_matrix,
%   end_to_end_y_derivative_matrix,local_x_derivative_matrix,
%   local_y_derivative_matrix)
%   uses inpaint_nans to interpolate discrete fibre angles over a continuum
%   both locally and for the end_to_end fibre. Angles are interpolated by
%   first interpolating the x and y derivatives, normalising them to unit 
%   vectors and then calculating the continuum angles. The method of spring
%   metaphors is used by inpaint_nans.
%
%   Input:
%   All values output from fibre_angle.m
%   end_to_end_x_derivative_matrix: matrix with discrete end to end 
%   x derivative values  
%   end_to_end_y_derivative_matrix: matrix with discrete end to end 
%   y derivative values  
%   local_x_derivative_matrix: matrix with discrete local x derivative 
%   values  
%   local_y_derivative_matrix: matrix with discrete local y derivative 
%   values  
%
%   Output:
%   end_to_end_angle_continuum: continuum interpolated matrix of end to end
%   angles
%   end_to_end_x_derivative_continuum: continuum interpolated matrix of 
%   end to end x derivatives
%   angles
%   end_to_end_y_derivative_continuum: continuum interpolated matrix of 
%   end to end x derivatives
%   local_angle_continuum: continuum interpolated matrix of local angles
%   local_x_derivative_continuum: continuum interpolated matrix of 
%   local x derivatives
%   local_y_derivative_continuum: continuum interpolated matrix of 
%   local y derivatives
%
%   Matrices are interpolated using inpaint_nans:
%   John D'Errico (2021). inpaint_nans 
%   (https://www.mathworks.com/matlabcentral/fileexchange/4551-inpaint_nans),
%   MATLAB Central File Exchange. Retrieved May 5, 2021.)
%
%   Class support for inputs end_to_end_x_derivative_matrix,
%   end_to_end_y_derivative_matrix,local_x_derivative_matrix,
%   local_y_derivative_matrix:
%      float: single, double
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.
end_to_end_x_derivative_continuum  = inpaint_nans(end_to_end_x_derivative_matrix,4); %Use spring metaphor method for inpainting
end_to_end_y_derivative_continuum  = inpaint_nans(end_to_end_y_derivative_matrix,4);
end_to_end_y_derivative_continuum(end_to_end_x_derivative_continuum<0)=-1*end_to_end_y_derivative_continuum(end_to_end_x_derivative_continuum<0);
end_to_end_x_derivative_continuum(end_to_end_x_derivative_continuum<0)=-1*end_to_end_x_derivative_continuum(end_to_end_x_derivative_continuum<0);
end_to_end_vector_magnitude=(end_to_end_x_derivative_continuum.^2+end_to_end_y_derivative_continuum.^2).^0.5;
end_to_end_x_derivative_continuum=end_to_end_x_derivative_continuum./end_to_end_vector_magnitude;
end_to_end_y_derivative_continuum=end_to_end_y_derivative_continuum./end_to_end_vector_magnitude;
end_to_end_derivative_ratio = end_to_end_y_derivative_continuum./((end_to_end_x_derivative_continuum.^2+end_to_end_y_derivative_continuum.^2).^0.5);
end_to_end_derivative_ratio((end_to_end_x_derivative_continuum==0)&(end_to_end_y_derivative_continuum==0))=1;%Where divide by zero set to 1 as numerator and denominator are same order of magnitude
end_to_end_angle_continuum = acos(end_to_end_derivative_ratio);

local_x_derivative_continuum  = inpaint_nans(local_x_derivative_matrix,4);
local_y_derivative_continuum  = inpaint_nans(local_y_derivative_matrix,4);
local_y_derivative_continuum(local_x_derivative_continuum<0)=-1*local_y_derivative_continuum(local_x_derivative_continuum<0);
local_x_derivative_continuum(local_x_derivative_continuum<0)=-1*local_x_derivative_continuum(local_x_derivative_continuum<0);
local_vector_magnitude=(local_x_derivative_continuum.^2+local_y_derivative_continuum.^2).^0.5;
local_x_derivative_continuum=local_x_derivative_continuum./local_vector_magnitude;
local_y_derivative_continuum=local_y_derivative_continuum./local_vector_magnitude;
local_derivative_ratio = local_y_derivative_continuum./((local_x_derivative_continuum.^2+local_y_derivative_continuum.^2).^0.5);
local_derivative_ratio((local_x_derivative_continuum==0)&(local_y_derivative_continuum==0))=1;%Where divide by zero set to 1 as numerator and denominator are same order of magnitude
local_angle_continuum = acos(local_derivative_ratio);   

end

