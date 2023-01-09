function [...
          area_weighted_sample,...
          neighbours,...
          variable_radius, ...
          inverted_bw_stats ...
          ] = circle_gap_statistics(...
          label_matrix,...
          radius_label_matrix,...
          centroid_row,...
          centroid_col,...
          circle_radius,...
          bw,...
          radii_vector...
          )

%CIRCLE_GAP_STATISTICS generates data to analyse the distribution of 
% circular gaps.
%
%[area_weighted_sample,neighbours,variable_radius,inverted_bw_stats]
%= circle_gap_statistics(label_matrix,radius_label_matrix,centroid_row,
% centroid_col,circle_radius,bw,radii_vector) analyses the output of 
% circle_gap_fitting() and generates distributions of radii of gaps
% neighbouring a given gap, distance between centroids of gaps, distance
% between boundaries of gaps and shape descriptors for gaps,
% 
%
%   Input:
%   label_matrix: Matrix of fitted circles with pixels corresponding to a
%   given circle all given a unique label value.
%   radius_label_matrix: Matrix of fitted circles with pixels corresponding
%   to radius of circle pizels are part of.
%   centroid_row: Vector of row locations of fitted circle centroids.
%   centroid_col: Vector of column locations of fitted circle centroids.
%   circle_radius: Vector of radii of all fitted circles.
%   bw: logical matrix of fibres (1) and background (0).
%   radii_vector: vector of minumum radii threshold used to determine 
%   centroid and boundary distances between gaps of larger radii.
%
%   Output:
%   area_weighted_sample: Distribution vector of circle radii weighted by 
%   area.
%   neighbour:  Matrix with dimensions equal to [size(bw),8]. Each
%   dimension corresponds to an 8 connected pixel neighbour. For each pixel
%   located within a gap the radii of gap occupying each of the 8 connected 
%   pixel neighbours is recoreded in these 8 dimensions. No gap in a given 
%   neighbour pixel is recorded with a value of zero. Pixel neighbours that
%   are from the same gap as the pixel in question are recorded with NaN 
%   values. The overall matrix then records the radius of gap neighouring 
%   and connected to each gap boundary.
%   variable_radius: A structure array with fields centroid_distance and 
%   boundary_distance. For each threshold radius in radii_vector it finds
%   all gaps with radius larger than the threshold and calculates the
%   pairwise distances between all gap centroids and pairwise minimum
%   distance between gap boundaries.
%   inverted_bw_stats: Records shape statistics from the regionprops 
%   commandfor the true gap shapes i.e. the shapes given by inverting the 
%   input logical fibre matrix, bw.
%
%
%   Class support for input bw:
%      logical
%   Class support for input label_matrix, radius_label_matrix, 
%   centroid_row, centroid_col, circle_radius 
%      float: single, double
%   Class support for input radii_vector
%      integer
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.


%Weight gaps by total area covered of each radius
area_weighted_sample = pi* radius_label_matrix(radius_label_matrix>0).^2;

%Find radii of all neighbouring gaps for each gap
neighbours=zeros([size(label_matrix),8]);
input_matrix = zeros([size(label_matrix)]);
input_matrix(2:end,2:end) = label_matrix(1:end-1,1:end-1);
input_matrix(input_matrix==label_matrix)=NaN;
neighbours(:,:,1) = input_matrix;
input_matrix = zeros([size(label_matrix)]);
input_matrix(2:end,:) = label_matrix(1:end-1,:);
input_matrix(input_matrix==label_matrix)=NaN;
neighbours(:,:,2)=input_matrix;
input_matrix = zeros([size(label_matrix)]);
input_matrix(2:end,1:end-1) = label_matrix(1:end-1,2:end);
input_matrix(input_matrix==label_matrix)=NaN;
neighbours(:,:,3)=input_matrix;
input_matrix = zeros([size(label_matrix)]);
input_matrix(:,2:end) = label_matrix(:,1:end-1);
input_matrix(input_matrix==label_matrix)=NaN;
neighbours(:,:,4)=input_matrix;
input_matrix = zeros([size(label_matrix)]);
input_matrix(:,1:end-1) = label_matrix(:,2:end);
input_matrix(input_matrix==label_matrix)=NaN;
neighbours(:,:,5)=input_matrix;
input_matrix = zeros([size(label_matrix)]);
input_matrix(1:end-1,2:end) = label_matrix(2:end,1:end-1);
input_matrix(input_matrix==label_matrix)=NaN;
neighbours(:,:,6)=input_matrix;
input_matrix = zeros([size(label_matrix)]);
input_matrix(1:end-1,:) = label_matrix(2:end,:);
input_matrix(input_matrix==label_matrix)=NaN;
neighbours(:,:,7)=input_matrix;
input_matrix = zeros([size(label_matrix)]);
input_matrix(1:end-1,1:end-1) = label_matrix(2:end,2:end);
input_matrix(input_matrix==label_matrix)=NaN;
neighbours(:,:,8)=input_matrix;

index_neighbours=find(neighbours>0);
labels=neighbours(index_neighbours);
neighbours(index_neighbours) = circle_radius(labels);

%Find distance between centroids and boundaries of all gaps over a radius
%given in radii_vector
sorted_circle_radius = sort(circle_radius,'descend');
for r=1:length(radii_vector)
    radius_distance_threshold = radii_vector(r);
    index = find(circle_radius>=radius_distance_threshold);
    if length(index)<5
        index = find(circle_radius>=sorted_circle_radius(min(5,length(sorted_circle_radius))));    
    end
    if length(index)>1
        neighbour_x = centroid_col(index);
        neighbour_y = centroid_row(index);
        centroid_coords=[neighbour_x',neighbour_y'];
        variable_radius(r).centroid_distance = pdist(centroid_coords);
    else
        variable_radius(r).centroid_distance = NaN;
    end


end
%Record shape statistics for gap shapes
L = bwlabel(~bw,8);
inverted_bw_stats = regionprops(...
    'table',...
    L,...
    'Area',...
    'Circularity',...
    'Eccentricity',...
    'EulerNumber',...
    'Extent',...
    'MajorAxisLength',...
    'MinorAxisLength',...
    'Perimeter'...
    );
end
