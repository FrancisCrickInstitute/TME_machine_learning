function gap_quadrant_analysis(variable_names,...
    output_stats,...
    tile_row,...
    tile_column,...
    total_rows,...
    total_columns,...
    save_directory,...
    ecm_mask,...
    tissue_mask,...
    discrete_gap_labels,...
    image_folder,...
    image_file,...
    image)
        
%GAP_QUADRANT_ANALYSIS generates gap feature values for sub-quadrants of 
% the original input tile. 
%
%gap_quadrant_analysis(variable_names,output_stats,tile_row,tile_column,
% total_rows,total_columns,save_directory,ecm_mask,tissue_mask,
% discrete_gap_labels,image) generates a csv of gap
% feature values for each quarter sub-quadrant. The features extracted are 
% analagous to the full-scale input image features (adjusted to account for
% the loss in scale). The function generates the aproppriate images and 
% masks  from the original input images and masks. If the proportion of 
% tissue is greater than 0.005 of the subtile then feature values are 
% generated and saved in a csv. 
% 
%
%   Input:

%   variable_names: Empty cell to populate with feature names.
%   output_stats: Empty array to populate with feature values.
%   tile_row: Gives row location of current sub-quadrant.
%   tile_column: Gives column location of current sub-quadrant.
%   total_rows: Total rows of full image.
%   total_columns: Total columns of full image.
%   save_directory: Directory to svae sub-quadrant csv output.
%   ecm_mask: Input mask of collagen locations for full image.
%   tissue_mask: Input mask of tissue locations for full image.
%   discrete_gap_labels: Input labels of discrete gaps for full image.
%   image_folder REMOVE
%   image_file REMOVE
%   image: Input grayscale deconvolved image for full image.
%
%   Output:
%   csv files named with a row-column convention.
%
%
%   Class support for input variable_names:
%      cell
%   Class support for input output_stats:
%      float: single, double
%   Class support for input ecm_mask, tissue_mask:
%      float: single, double
%   Class support for input tile_row, tile_column, total_rows, 
%   total_columns, discrete_gap_labels,image
%      integer
%   Class support for input save_directory:
%      string
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.





row_lb = tile_row * round(total_rows/2) + 1;
row_ub = min((tile_row+1) * round(total_rows/2),total_rows);
column_lb = tile_column * round(total_columns/2) + 1;
column_ub = min((tile_column+1) * round(total_columns/2),total_columns);

discrete_gap_labels = discrete_gap_labels(row_lb:row_ub,column_lb:column_ub);
unique_discrete_gap_labels=unique(discrete_gap_labels);
unique_discrete_gap_labels=unique_discrete_gap_labels(unique_discrete_gap_labels>0);

for relabel_index=1:length(unique_discrete_gap_labels)
    object_index=find(discrete_gap_labels==unique_discrete_gap_labels(relabel_index));
    discrete_gap_labels(object_index)=relabel_index;
end
tissue_mask = logical(tissue_mask(row_lb:row_ub,column_lb:column_ub));
ecm_mask = logical(ecm_mask(row_lb:row_ub,column_lb:column_ub));
image = image(row_lb:row_ub,column_lb:column_ub);
tissue_proportion = sum(tissue_mask(:))/((row_ub-row_lb+1)*(column_ub-column_lb+1));

if tissue_proportion > 0.005
    input_gap_matrix = ecm_mask;
    inverted_gap_matrix = ~input_gap_matrix.*tissue_mask;
    no_holes_matrix = logical(max(~tissue_mask,input_gap_matrix));

    [...
      label_matrix,...
      radius_label_matrix,...
      centroid_row,...
      centroid_col,...
      circle_radius...
    ] = circle_gap_fitting(no_holes_matrix ,2);
    radii_vector = [prctile(circle_radius,90),prctile(circle_radius,95),prctile(circle_radius,99),prctile(circle_radius,99.9)]; 
    [...
      area_weighted_sample,...
      neighbours,...
      variable_radius ...
      ] = circle_gap_statistics(label_matrix,radius_label_matrix,centroid_row,centroid_col,circle_radius,input_gap_matrix,radii_vector);

    output_stats = [output_stats,length(circle_radius)/sum(tissue_mask(:))];
    variable_names={variable_names{1:end},['total_circles']};
    output_stats = [output_stats,mean(circle_radius),std(circle_radius),skewness(circle_radius),kurtosis(circle_radius)];
    variable_names={variable_names{1:end},'mean_gap_radius','std_gap_radius','skewness_gap_radius','kurtosis_gap_radius'};
    percentiles=[90,95,99,99.9,99.99,100];
    for I = 1:length(percentiles)
        p = percentiles(I);
        output_stats = [output_stats,prctile(circle_radius,p)];
        variable_names={variable_names{1:end},['gap_size_percentile_' num2str(p) ]};
    end

    percentiles=[5,10,25,50,75,90,95,99,100];
    output_stats = [output_stats,mean(area_weighted_sample),std(area_weighted_sample),skewness(area_weighted_sample),kurtosis(area_weighted_sample)];
    variable_names={variable_names{1:end},'mean_area_weighted_gap_radius','std_area_weighted_gap_radius','skewness_area_weighted_gap_radius','kurtosis_area_weighted_gap_radius'};
    for I = 1:length(percentiles)
        p = percentiles(I);
        output_stats = [output_stats,prctile(area_weighted_sample,p)];
        variable_names={variable_names{1:end},['area_weighted_gap_size_percentile_' num2str(p) ]};
    end


    output_stats = [output_stats,mean(neighbours(:),'omitnan'),std(neighbours(:),'omitnan'),skewness(neighbours(:)),kurtosis(neighbours(:))];
    variable_names={variable_names{1:end},'mean_gap_neighbour_size','std_gap_neighbour_size','skewness_gap_neighbour_size','kurtosis_gap_neighbour_size'};

    
    prctile_radii_vector={'90','95','99','99_9'};
    for r=1:length(radii_vector)
        output_stats = [output_stats,mean(variable_radius(r).centroid_distance),std(variable_radius(r).centroid_distance),skewness(variable_radius(r).centroid_distance),kurtosis(variable_radius(r).centroid_distance)];
        variable_names={variable_names{1:end},['mean_gap_centroid_distance_radius_threshold_'  prctile_radii_vector{r}],['std_gap_centroid_distance_radius_threshold_' prctile_radii_vector{r}],['skewness_gap_centroid_distance_radius_threshold_' prctile_radii_vector{r}],['kurtosis_gap_centroid_distance_radius_threshold_' prctile_radii_vector{r}]};
        
    end
    
    discrete_gap_statistics = regionprops('table',discrete_gap_labels,'Area','Circularity','ConvexArea','Eccentricity','EquivDiameter','EulerNumber','Extent','FilledArea','MajorAxisLength','MinorAxisLength','Perimeter','Solidity');
    if height(discrete_gap_statistics) ==0
        for I=1:width(discrete_gap_statistics)
                shape_stat = discrete_gap_statistics.Variables;
                shape_stat = shape_stat(:,I);
                shape_stat(isinf(shape_stat))=[];
                shape_stat(isnan(shape_stat))=[];
                stat_string = discrete_gap_statistics.Properties.VariableNames{I}
                output_stats = [output_stats,0,0,NaN,NaN];
                variable_names={variable_names{1:end},['mean_gap_shape_' stat_string],['std_gap_shape_' stat_string],['skewness_gap_shape_' stat_string]};
        end
    else
        for I=1:width(discrete_gap_statistics)
            shape_stat = discrete_gap_statistics.Variables;
            shape_stat = shape_stat(:,I);
            shape_stat(isinf(shape_stat))=[];
            shape_stat(isnan(shape_stat))=[];
            stat_string = discrete_gap_statistics.Properties.VariableNames{I};
            k = strfind(stat_string,'Area');
            if isempty(k)==0
                shape_stat = shape_stat./length(tissue_mask(:));
            end
            output_stats = [output_stats,mean(shape_stat),std(shape_stat),skewness(shape_stat),kurtosis(shape_stat)];
            variable_names={variable_names{1:end},['mean_gap_shape_' stat_string],['std_gap_shape_' stat_string],['skewness_gap_shape_' stat_string],['kurtosis_gap_shape_' stat_string]};
        end
        for I=1:width(discrete_gap_statistics)
            shape_stat = discrete_gap_statistics.Variables;
            area = shape_stat(:,1);
            shape_stat = shape_stat(:,I);
            area(isinf(shape_stat))=[];
            area(isnan(shape_stat))=[];
            shape_stat(isinf(shape_stat))=[];
            shape_stat(isnan(shape_stat))=[];
            shape_stat(isinf(area))=[];
            shape_stat(isnan(area))=[];
            area(isinf(area))=[];
            area(isnan(area))=[];
            stat_string = discrete_gap_statistics.Properties.VariableNames{I};
            k = strfind(stat_string,'Area');
            if isempty(k)==0
                shape_stat = shape_stat./length(tissue_mask(:));
            end
            area_weighted_data = repelem(shape_stat,area);
            output_stats = [output_stats,mean(area_weighted_data),std(area_weighted_data),skewness(area_weighted_data),kurtosis(area_weighted_data)];
            variable_names={variable_names{1:end},['area_weighted_mean_gap_shape_' stat_string],['area_weighted_std_gap_shape_' stat_string],['area_weighted_skewness_gap_shape_' stat_string],['area_weighted_kurtosis_gap_shape_' stat_string]};
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Save data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    csv_input=[variable_names;num2cell(output_stats)];
    csv_name=[save_directory,'gap_features_out_',num2str(tile_row,'%.2d'),'_',num2str(tile_column,'%.2d'),'.csv'];   
    writecell(csv_input',csv_name);

end

end