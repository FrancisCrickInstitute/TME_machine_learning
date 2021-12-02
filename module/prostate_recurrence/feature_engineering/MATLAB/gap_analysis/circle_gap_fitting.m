function [...
          label_matrix,...
          radius_label_matrix,...
          centroid_row,...
          centroid_col,...
          circle_radius...
          ] = circle_gap_fitting(bw,minimum_radius)

%CIRCLE_GAP_FITTING fits the maximum sized circles that fit between gaps in
%a logical image of background (0) and foreground (1).
%
%[label_matrix,radius_label_matrix,centroid_row,centroid_col,circle_radius]
%= circle_gap_fitting(bw,minimum_radius) takes a logical matrix and fits the
%maximum circles that fit between gaps. An input logical matrix is padded 
%such that edges also count as fibres when calculated gap sizes. The 
%distance output is calculated from gap_distance_function. The largest 
%circle is recorded (randomly for ties) and the distance output then 
%updated accounting for further circles now being unable to overlap with 
%this first recorded circle in addition to fibres and the padded edges. The 
%process then iterates until no more gaps remain to be filled. 
%
%
%   Input:
%   bw: Logical matrix of fibres and background.
%   minimum_radius: The minimum radius of circles to be recorded.
%
%   Output:
%   label_matrix: Matrix of fitted circles with pixels corresponding to a
%   given circle all given a unique label value.
%   radius_label_matrix: Matrix of fitted circles with pixels corresponding
%   to radius of circle pizels are part of.
%   centroid_row: Vector of row locations of fitted circle centroids.
%   centroid_col: Vector of column locations of fitted circle centroids.
%   circle_radius: Vector of radii of all fitted circles.

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

%Anonymous function to generate filled circles of a given radius.
circle_function = @(rows,row_centre,cols,column_centre,radius)...
    (rows - row_centre).^2 + (cols - column_centre).^2 <= radius.^2;

[row_size,col_size]=size(bw);
binary_padded = zeros(row_size+2,col_size+2);
binary_padded(2:row_size+1,2:col_size+1)=bw;
padded_size = size(binary_padded);
[...
irows,...
icols,...
bw_analyse,...
distance_matrix,...
radius_matrix,...
radius_vector,...
radius_index_vector,...
total_circles...
] = gap_distance_function(binary_padded,minimum_radius);

counter=0;
label_matrix = zeros(padded_size);
radius_label_matrix = zeros(padded_size);
while isempty(radius_vector)==0
    counter=counter+1;
    
    max_radius_index=find(radius_vector==max(radius_vector));
    random_circle_index=max_radius_index(randi(length(max_radius_index)));
    [row_mid,col_mid] = ...
        ind2sub(padded_size,radius_index_vector(random_circle_index));
    [cols rows] = meshgrid(1:icols, 1:irows);
    
    
    
    circle_radius_float = radius_vector(random_circle_index);
    radius = circle_radius_float-1;
    upper_radius = ceil(radius);
    lower_radius = floor(radius);
    
    %For non-integer distances we first check whether the rounded value to 
    %positive infinity overlaps with foreground. If is does we then take
    %the rounded value to negative infinity as radius.
    if upper_radius~=lower_radius
        radius = upper_radius;
        circlePixels = circle_function(rows,row_mid,cols,col_mid,radius);
        %circlePixels = (rows - row_mid).^2 + (cols - col_mid).^2 <= radius.^2;
        if max(bw_analyse(circlePixels))==1
           radius = lower_radius;
           circlePixels = ...
               circle_function(rows,row_mid,cols,col_mid,radius);
        end
    else
        circlePixels = ...
            circle_function(rows,row_mid,cols,col_mid,radius);
    end
    bw_analyse(circlePixels)=1;
    centroid_row(counter)=row_mid;
    centroid_col(counter)=col_mid;
    circle_radius(counter)=radius+0.5; %Added 0.5 to account for a single pixel having radius 0.5
    %Should we take the discrete or continuous version of this?
    label_matrix(circlePixels) = counter;
    radius_label_matrix(circlePixels) = radius + 0.5;
    [...
    irows,...
    icols,...
    bw_analyse,...
    distance_matrix,...
    radius_matrix,...
    radius_vector,...
    radius_index_vector,...
    total_circles...
    ] = gap_distance_function(bw_analyse,minimum_radius);
    %proportioncomplete=1-length(radius_vector)/(irows*icols)%Timer
end

%Output data updated to account for original padding.
label_matrix=label_matrix(2:row_size+1,2:col_size+1);
radius_label_matrix=radius_label_matrix(2:row_size+1,2:col_size+1);
centroid_row=centroid_row-1;
centroid_col=centroid_col-1;

end


    
