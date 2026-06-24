function feature_extraction_hpc_pipeline_gaps(image_start,image_end,image_folder,tissue_folder,function_folder,ctfire_function_folder,ecm_threshold)

addpath(genpath(function_folder))
image_type = '.tif';
input_image_dir = [image_folder '*' image_type] ;
input_image_list = dir(input_image_dir);

output_filetype='tif';
for image_I=image_start:image_end
    variable_names={};
    output_stats = [];
    image_file = input_image_list(image_I).name;
    image = imread([image_folder '/' image_file]);

    [~, name, ext] = fileparts([image_folder image_file]);
    tissue_name = name(1:end-4);
    tissue_mask = imread([tissue_folder,tissue_name,ext]);
    tissue_mask = logical(tissue_mask);
    tissue_object_label = bwlabel(tissue_mask,8);
    unique_tissue_objects = unique(tissue_object_label(:));
    if max(unique_tissue_objects>0)
        unique_tissue_objects =unique_tissue_objects(unique_tissue_objects>0);
        for L=1:length(unique_tissue_objects)
            object_L=unique_tissue_objects(L);
            index_object = find(tissue_object_label==object_L);
            if max(image(index_object))==0
                tissue_mask(index_object)=0;
            end
        end
    end
    [row_dim, col_dim] = size(image);

    ecm_mask = image;
    ecm_mask(ecm_mask<=ecm_threshold)=0;
    ecm_mask(ecm_mask>ecm_threshold)=1;
    ecm_mask = logical(logical(ecm_mask).*tissue_mask);
    ecm_mask = bwareaopen(ecm_mask,20,8);

  %  if sum(ecm_mask(:))/length(tissue_mask(:)) >= 0.005%Threshold to accept tile
        %Run CT-Fire
        
        image_folder
        output_folder = [strrep(image_folder,'pre_processed_data','feature_engineering'),'tile_level_features/',name '/'];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Gap Analysis
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        
        inverted_gap_matrix = ~ecm_mask.*tissue_mask;
        no_holes_matrix = logical(max(~tissue_mask,ecm_mask));


        output_folder = [strrep(image_folder,'pre_processed_data','feature_engineering'),'tile_level_features/',name '/'];
        str1 = strfind(output_folder,'/');
        str2 = strfind(output_folder,'feature_engineering/');
        str_index=str1(str1>str2);
        for str_i=1:length(str_index)
           directory_create=output_folder(1:str_index(str_i))
           mkdir(directory_create)
        end
        output_image_dir = [output_folder '*gap*' image_type];
        delete(output_image_dir)
        output_csv_dir = [output_folder 'gap' '*.csv' ];
        delete(output_csv_dir)
        output_mat_dir = [output_folder 'gap' '*.mat' ];
        delete(output_mat_dir)

        save_directory = [output_folder '/' name '_automated_results/'];
        save_directory = output_folder;
        mkdir(save_directory);


        [...
          label_matrix,...
          radius_label_matrix,...
          centroid_row,...
          centroid_col,...
          circle_radius...
        ] = circle_gap_fitting(no_holes_matrix ,2);
        ImName=[save_directory 'circle_gaps.tif']
        circle_gap_plotting(label_matrix,ecm_mask,ImName);
        radii_vector = [prctile(circle_radius,90),prctile(circle_radius,95),prctile(circle_radius,99),prctile(circle_radius,99.9)]; 
        [...
          area_weighted_circles,...
          neighbours,...
          variable_radius ...
          ] = circle_gap_statistics(label_matrix,radius_label_matrix,centroid_row,centroid_col,circle_radius,ecm_mask,radii_vector);
        area_weighted_circles=(area_weighted_circles.^0.5)./pi;


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
        output_stats = [output_stats,mean(area_weighted_circles),std(area_weighted_circles),skewness(area_weighted_circles),kurtosis(area_weighted_circles)];
        variable_names={variable_names{1:end},'mean_area_weighted_gap_radius','std_area_weighted_gap_radius','skewness_area_weighted_gap_radius','kurtosis_area_weighted_gap_radius'};
        for I = 1:length(percentiles)
            p = percentiles(I);
            output_stats = [output_stats,prctile(area_weighted_circles,p)];
            variable_names={variable_names{1:end},['area_weighted_gap_size_percentile_' num2str(p) ]};
        end


        output_stats = [output_stats,mean(neighbours(:),'omitnan'),std(neighbours(:),'omitnan'),skewness(neighbours(:)),kurtosis(neighbours(:))];
        variable_names={variable_names{1:end},'mean_gap_neighbour_size','std_gap_neighbour_size','skewness_gap_neighbour_size','kurtosis_gap_neighbour_size'};

        
        prctile_radii_vector={'90','95','99','99_9'};
        for r=1:length(radii_vector)
            output_stats = [output_stats,mean(variable_radius(r).centroid_distance),std(variable_radius(r).centroid_distance),skewness(variable_radius(r).centroid_distance),kurtosis(variable_radius(r).centroid_distance)];
            variable_names={variable_names{1:end},['mean_gap_centroid_distance_radius_threshold_'  prctile_radii_vector{r}],['std_gap_centroid_distance_radius_threshold_' prctile_radii_vector{r}],['skewness_gap_centroid_distance_radius_threshold_' prctile_radii_vector{r}],['kurtosis_gap_centroid_distance_radius_threshold_' prctile_radii_vector{r}]};
            
        end

        discrete_gap_labels = discrete_gap_extractor(ecm_mask,tissue_mask,image);
        unique_discrete_gap_labels=unique(discrete_gap_labels);
        unique_discrete_gap_labels=unique_discrete_gap_labels(unique_discrete_gap_labels>0);
        ImName=[save_directory 'discrete_gaps.tif']
        circle_gap_plotting(discrete_gap_labels,ecm_mask,ImName);

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
                area_weighted_discrete_gaps = repelem(shape_stat,area);
                output_stats = [output_stats,mean(area_weighted_discrete_gaps),std(area_weighted_discrete_gaps),skewness(area_weighted_discrete_gaps),kurtosis(area_weighted_discrete_gaps)];
                variable_names={variable_names{1:end},['area_weighted_mean_gap_shape_' stat_string],['area_weighted_std_gap_shape_' stat_string],['area_weighted_skewness_gap_shape_' stat_string],['area_weighted_kurtosis_gap_shape_' stat_string]};
           end
            
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Save data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        csv_input=[variable_names;num2cell(output_stats)];
        csv_name=[save_directory,'gap_features_out.csv'];   
        writecell(csv_input',csv_name);
        matlab_name=[save_directory,'gap_features_out.mat'];
        clearvars -except variable_names output_stats tile_row tile_column...
        row_dim col_dim save_directory ecm_mask tissue_mask discrete_gap_labels...
        image_folder image_file image centroid_col centroid_row circle_radius...
        label_matrix radius_label_matrix matlab_name


        save(matlab_name);


        for tile_row = 0:1
            for tile_column = 0:1
                variable_names={};
                output_stats = [];
                gap_quadrant_analysis(variable_names,output_stats,tile_row,tile_column,row_dim,col_dim,save_directory,ecm_mask,tissue_mask,discrete_gap_labels,image_folder,image_file,image);
            end
        end
%     else
%         'The ecm proportion is below threshold. No fibres can be extracted and there will be one large gap covering the whole tile.'
%     end

    
end
end




