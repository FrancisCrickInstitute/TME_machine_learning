function [angle_difference,weighting_ecm] = ...
    random_alignment_quantification(...
    radial_distance,...
    col_dim,...
    row_dim,...
    ecm_index,...
    ecm_random_sample,...
    image,...
    local_angle_continuum)

%RANDOM_ALIGNMENT_QUANTIFICATION generates a distribution of differences in
% angle for a sample of array elements and their neighbours a given length-
% scale away.
%
%[angle_difference,weighting_ecm] = random_alignment_quantification(
% radial_distance,col_dim,row_dim,ecm_index,ecm_random_sample,image,
% local_angle_continuum) takes a reference sample of array indices where 
% collagen is present and for a given length-scale (i.e. radial distance) 
% finds the indices of all neighbouring points that distance away and 
% located where the ecm mask exists. It then outputs the absolute 
% difference in angle between the sample of reference points and their 
% neighbours, binning into a single sample. The pixel intensity at each 
% neighbouring point is also recorded and binned to allow for ecm density 
% weighted alignment quantification.
% 
%
%   Input:
%   radial_distance: The defined radial distance to search for neighbours
%   of each sample point.
%   col_dim: Column dimension of input array.
%   row_dim: Row dimension of input array.
%   ecm_index: Indices of locations where ecm_mask is foreground.
%   ecm_random_sample: Indices of input sample of reference points.
%   Typically randomly sampled.
%   image: Input deconvolved greyscale image. Used to provide pixel
%   intensity values to give ecm density at each location.
%   local_angle_continuum: Array of fibre angles (continuum rather than 
%   discrete). 
%
%   Output:
%   angle_difference: Distribution vector of binned difference in angle
%   between all reference sample angles and each of their neighbour angles.
%   area.
%   weighting_ecm: pixel intensity (ecm density) for each point in the 
%   angle_difference vector used for weighting angles by density. 
%
%
%   Class support for radial_distance, col_dim, row_dim, ecm_index, 
%   ecm_random_sample, image:
%      integer
%
%   Class support for input local_angle_continuum:
%      float: single, double
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.



[rows,columns] = ind2sub([row_dim,col_dim],ecm_random_sample);
SE = strel('disk',radial_distance,0);
circle_array = zeros([row_dim,col_dim]);
circle_array(round(row_dim/2),round(col_dim/2))=1;
circle_array = imdilate(circle_array,SE);
circle_array =bwmorph(circle_array,'remove');
circle_index = find(circle_array);
[circle_row,circle_col] = ind2sub(size(circle_array),circle_index);
[rows_mesh,circle_rows_mesh] = meshgrid(rows,circle_row);
row_indices = rows_mesh+circle_rows_mesh-round(row_dim/2);
[columns_mesh,circle_columns_mesh] = meshgrid(columns,circle_col);
column_indices = columns_mesh+circle_columns_mesh-round(col_dim/2);


rows_mesh_indices=rows_mesh(:);
columns_mesh_indices=columns_mesh(:);

column_indices=column_indices(:);
row_indices=row_indices(:);

column_indices(row_indices<1)=[];
row_indices(row_indices<1)=[];
rows_mesh_indices(row_indices<1)=[];
columns_mesh_indices(row_indices<1)=[];
column_indices(row_indices>row_dim)=[];
row_indices(row_indices>row_dim)=[];
rows_mesh_indices(row_indices>row_dim)=[];
columns_mesh_indices(row_indices>row_dim)=[];

row_indices(column_indices<1)=[];
column_indices(column_indices<1)=[];
rows_mesh_indices(column_indices<1)=[];
columns_mesh_indices(column_indices<1)=[];
row_indices(column_indices>col_dim)=[];
column_indices(column_indices>col_dim)=[];
rows_mesh_indices(column_indices>col_dim)=[];
columns_mesh_indices(column_indices>col_dim)=[];


all_neighbour_index = sub2ind(size(circle_array),...
                              row_indices,...
                              column_indices);
reference_pixel_index = sub2ind(size(circle_array),...
                                rows_mesh_indices,...
                                columns_mesh_indices);

all_neighbour_in_mask_index = ismember(all_neighbour_index,ecm_index);
all_neighbour_index=all_neighbour_index(all_neighbour_in_mask_index);
reference_pixel_index=reference_pixel_index(all_neighbour_in_mask_index);

weighting_ecm = double(image(all_neighbour_index));
neighbour_angle = local_angle_continuum(all_neighbour_index);
reference_angle=local_angle_continuum(reference_pixel_index);
angle_difference = abs(neighbour_angle - reference_angle);

end









