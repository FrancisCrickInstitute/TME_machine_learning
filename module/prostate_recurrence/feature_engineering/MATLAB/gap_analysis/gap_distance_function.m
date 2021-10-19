function [...
          irows,...
          icols,...
          bw_bounded,...
          distance_matrix,...
          radius_matrix,...
          radius_vector,...
          radius_index_vector,...
          total_circles...
          ]...
          = gap_distance_function(bw,minimum_radius)
%GAP_DISTANCE_FUNCTION calculates the minimum distance for each background 
% pixel and the nearest fibre for a logical input matrix of fibres and
% produces related output data.
%
%    [irows,icols,radius_vector,radius_index_vector,radius_matrix,
%    total_circles,bw,distance_matrix]
%         = gap_distance_function(bw,minimum_radius) reads in a logical 
%    input matrix of fibres (=1) and background (=0). It adds a boundary of
%    foreground pixels around the image abd then calculates the distance 
%    transform of all background pixels to the nearest foreground pixel. 
%    Distances below a user defined threshold are set to zero. The radii
%    and corresponding index location are recorded. 
%
%   Input:
%   bw: Logical matrix of fibres and background.
%   minimum_radius: The minimum radius to be recorded.
%
%   Output:
%   irows: Number of rows of input matrix.
%   icols: Number of columns of input matrix.
%   bw_bounded: Input logical matrix with boundary added to image edges.
%   distance_matrix: Distance transform of bw_bounded.
%   radius_matrix: Thresholded Distance transform of bw_bounded with 
%   distances below minimum_radius set to zero.
%   radius_vector: Vector of all non-zero distances.
%   radius_index_vector: Vector of matrix index locations of radius_vector.
%   total_circles: Total number of pixels with non-zero distance to 
%   background.
%
%
%   Class support for input bw:
%      logical
%   Class support for input minimum_radius:
%      float: single, double
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.
    [irows,icols]=size(bw);
    bw_bounded = bw;
    bw_bounded(1,:)=1;
    bw_bounded(irows,:)=1;
    bw_bounded(:,1)=1;
    bw_bounded(:,icols)=1;
    [distance_matrix,~] = bwdist(bw_bounded);
    distance_matrix(distance_matrix<minimum_radius) = 0;

    radius_vector=double(distance_matrix(:));
    radius_index_vector = find(radius_vector);
    radius_vector = radius_vector(radius_vector>0);   
    
    radius_matrix=distance_matrix;
    radius_matrix(radius_matrix<minimum_radius)=0;
        
    total_circles = length(radius_vector);
    
end