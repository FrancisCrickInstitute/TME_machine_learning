function [angle_difference,weighting_ecm] = random_alignment_quantification(...
          radial_distance,...
          col_dim,...
          row_dim,...
          ecm_index,...
          ecm_random_sample,...
          image,...
          local_angle_continuum)
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


all_neighbour_index = sub2ind(size(circle_array),row_indices,column_indices);
reference_pixel_index = sub2ind(size(circle_array),rows_mesh_indices,columns_mesh_indices);

all_neighbour_in_mask_index = ismember(all_neighbour_index,ecm_index);
all_neighbour_index=all_neighbour_index(all_neighbour_in_mask_index);
reference_pixel_index=reference_pixel_index(all_neighbour_in_mask_index);

weighting_ecm = double(image(all_neighbour_index));
neighbour_angle = local_angle_continuum(all_neighbour_index);
reference_angle=local_angle_continuum(reference_pixel_index);
angle_difference = abs(neighbour_angle - reference_angle);

end
















% radii = round([0.5,1,2,5,10,15,20,30,40,50,60,80,100]/0.22);
% ecm_mask;
% [row_dim,col_dim]=size(ecm_mask);
% sample_size = 1000;
% radial_distances=[10,20,100];
% ecm_index=find(ecm_mask);
% rand_perm_index = randperm(length(ecm_index),sample_size);
% 
% 
% 
% [rows,columns] = ind2sub(size(ecm_mask),ecm_index(rand_perm_index));
% for r=1:length(radial_distances)
%     radial_distance=radial_distances(r);
%     SE = strel('disk',radial_distance,0);
%     circle_array = zeros([row_dim,col_dim]);
%     circle_array(round(row_dim/2),round(col_dim/2))=1;
%     circle_array = imdilate(circle_array,SE);
%     circle_array =bwmorph(circle_array,'remove');
%     circle_index = find(circle_array);
%     [circle_row,circle_col] = ind2sub(size(circle_array),circle_index);
%     [rows_mesh,circle_rows_mesh] = meshgrid(rows,circle_row);
%     row_indices = rows_mesh+circle_rows_mesh-round(row_dim/2);
%     [columns_mesh,circle_columns_mesh] = meshgrid(columns,circle_col);
%     column_indices = columns_mesh+circle_columns_mesh-round(col_dim/2);
%     
%     
%     rows_mesh_indices=rows_mesh(:);
%     columns_mesh_indices=columns_mesh(:);
% 
%     column_indices=column_indices(:);
%     row_indices=row_indices(:);
%     
%     column_indices(row_indices<1)=[];
%     row_indices(row_indices<1)=[];
%     rows_mesh_indices(row_indices<1)=[];
%     columns_mesh_indices(row_indices<1)=[];
%     column_indices(row_indices>row_dim)=[];
%     row_indices(row_indices>row_dim)=[];
%     rows_mesh_indices(row_indices>row_dim)=[];
%     columns_mesh_indices(row_indices>row_dim)=[];
% 
%     row_indices(column_indices<1)=[];
%     column_indices(column_indices<1)=[];
%     rows_mesh_indices(column_indices<1)=[];
%     columns_mesh_indices(column_indices<1)=[];
%     row_indices(column_indices>col_dim)=[];
%     column_indices(column_indices>col_dim)=[];
%     rows_mesh_indices(column_indices>col_dim)=[];
%     columns_mesh_indices(column_indices>col_dim)=[];
% 
% 
%     all_neighbour_index = sub2ind(size(circle_array),row_indices,column_indices);
%     reference_pixel_index = sub2ind(size(circle_array),rows_mesh_indices,columns_mesh_indices);
% 
%     all_neighbour_in_mask_index = ismember(all_neighbour_index,ecm_index);
%     all_neighbour_index=all_neighbour_index(all_neighbour_in_mask_index);
%     reference_pixel_index=reference_pixel_index(all_neighbour_in_mask_index);
% 
%     weighting_ecm = image(all_neighbour_index);
%     neighbour_angle = local_angle_continuum(all_neighbour_index);
%     reference_angle=local_angle_continuum(reference_pixel_index);
%     angle_difference = abs(neighbour_angle - reference_angle);
% %     figure
% %     histogram(angle_difference)
% % end
% % 
% % 
% % 
% % 
% % scatter(column_indices(:),row_indices(:),'g.')
% % local_angle_continuum,radius_input','tissue_mask'