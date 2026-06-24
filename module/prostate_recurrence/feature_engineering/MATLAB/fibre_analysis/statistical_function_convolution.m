function [output_image,output_length] = ...
    statistical_function_convolution(...
    input_continuum, ...
    radius,...
    stats_function,...
    logical_boundary,...
    type,...
    tissue_mask,...
    disk_directory,...
    approximate_disk...
    )
%STATISTICAL_FUNCTION_CONVOLUTION carries out convolutions of statistical
%functions on continuum matrices.
%
%   [output_image,output_length] = 
%   statistical_function_convolution(input_continuum,radius,stats_function,
%   logical_boundary,type)
%   carries out a convolution of a continuum matrix using one of eleven 
%   statistical functions. The convolution filter is either all points 
%   within a circle of given radius or all points on the boundary. 
%   The convolution is applied either just in terms of those points or in 
%   terms of a given releationship between those points and the point in 
%   the continuum matrix currently being convolved. 
%
%   The convolution filter applied to each point on the input_continuum is
%   applied to a circle of a user determined radius. The convolution filter
%   can be designed to only consider points on the circle boundary or all 
%   points in the circle. The convolution can additionally be applied in
%   terms of the relationship between these points and the central point of
%   the circle (i.e. the point in input_continuum currently being
%   convoluted). This relationship can be in the form of absolute
%   difference between convoluted point and all other points or ratio
%   between convoluted point and all other points.
% 
%   To account for boundary effects on the convolution filter, a new 
%   periodic matrix is generated that is nine times larger than the current
%   matrix. The matrix is then downsized to the minimum size required by 
%   the chosen circle radius to increas convolution speed.  
%
%   Input:
%   input_continuum: Matrix of continuum field where convolution will be
%   applied.
%   radius: Radius of circle used for convolution. The maximum radius is
%   the minimum of half the number of rows and columns of the matrix i.e.
%   the maximum circle size allowed is one that fits entirely within the
%   image.
%   stats_function: The statistical function used in the convolution 
%   operation. Choices are median, mean, sum, variance, standard deviation,
%   minimum, maximum, lower quartile, upper quartile, skewness and
%   kurtosis.
%   logical_boundary: Whether only points on the boundary 
%   (logical_boundary=1) or all points in the whole circle 
%   (logical_boundary=0) are used for the convolution.
%   type: The relationship between the points selected within the circle to
%   the centre point of the circle for convolution purposes.
%   0: statistical function applied to all points in the selected region
%   (circle boundary or whole circle)
%   1: statistical function applied to the absolute difference between the
%   convoluted point (centre of the circle) and selected region points.
%   This function specifically is for use with differences in angle so that
%   all values fall in the range [0,pi]
%   2: Statistical function applied to ratio of convoluted point and
%   selected region points. 
%   3: Statistical function applied to absolute difference statistical 
%   function applied to the absolute difference between the
%   convoluted point (centre of the circle) and selected region points.
%   This function should be used when input is not angles
%
%   output_image: Output convolved image, same dimensions as input image
%   output_length: Total number of neighbouring points used in convolution.
%
%
%   Class support for input input_continuum:
%      float: single, double
%   Class support for input radius, logical_boundary, type:
%      int: uint8, uint16, uint64
%   Class support for input stats_function:
%      string
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.

if approximate_disk == 0
    file_name = ['radius_', num2str(radius), '_true_circle.mat'];
else
    file_name = ['radius_', num2str(radius), '_approximate_circle.mat'];
end
disk_name = fullfile(disk_directory,file_name); 
load(disk_name);

input_continuum(~tissue_mask)=NaN;
analyzed_image = padarray(input_continuum, [radius+1 radius+1],'symmetric'); 

switch logical_boundary
    case 1
        boundary_location=outer_boundary;
        output_length = length_boundary;
    case 0
        boundary_location=inner_region;
        output_length = length_inner;
end

switch type
    case 0
        input_vector = 'x(boundary_location)';
    case 1
        input_vector = ...
            'subtract_angles(x(centre_index),x(boundary_location))';
    case 2
        input_vector = ...
            'ratio_angles(x(centre_index),x(boundary_location))';
    case 3
        input_vector = 'x(centre_index)-x(boundary_location)';
end

stats_function = lower(stats_function);
switch stats_function
    case{'median'}
        convolution_function = ['@(x) median(' input_vector ',''omitnan'')'];
    case{'mean'}
        convolution_function = ['@(x) mean(' input_vector ',''omitnan'')'];
    case{'sum'}
        convolution_function = ['@(x) sum(' input_vector ',''omitnan'')'];
    case{'var'}
        convolution_function = ['@(x) var(' input_vector ',''omitnan'')'];
    case{'std'}
        convolution_function = ['@(x) std(' input_vector ',''omitnan'')'];
    case{'min'}
        convolution_function = ['@(x) min(' input_vector ',[],''omitnan'')'];
    case{'max'}
        convolution_function = ['@(x) max(' input_vector ',[],''omitnan'')'];
    case{'lower quartile'}
        convolution_function = ['@(x) prctile(' input_vector ',25)'];
    case{'upper quartile'}
        convolution_function = ['@(x) prctile(' input_vector ',75)'];
    case{'skewness'}
        convolution_function = ['@(x) skewness(' input_vector ')'];
    case{'kurtosis'}
        convolution_function = ['@(x) kurtosis(' input_vector ')'];
end
convolution_image = ...
    nlfilter(...
    analyzed_image,...
    [filtersize filtersize],...
    eval(convolution_function)...
    );
[rows,cols] = size(input_continuum);
output_image = ...
    convolution_image(radius+1:rows+radius,radius+1:cols+radius);
end

