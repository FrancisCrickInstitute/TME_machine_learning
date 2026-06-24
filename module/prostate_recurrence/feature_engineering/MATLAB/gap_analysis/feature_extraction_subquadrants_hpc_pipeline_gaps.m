function feature_extraction_subquadrants_hpc_pipeline_gaps(image_start,image_end,image_folder,function_folder)

addpath(genpath(function_folder))
image_type = '.tif';
input_image_dir = [image_folder '*' image_type] ;
input_image_list = dir(input_image_dir);

current_dir = pwd;
output_filetype='tif';
for image_I=image_start:image_end
    image_file = input_image_list(image_I).name;
    [~, name, ext] = fileparts([image_folder image_file]);
    output_folder = [strrep(image_folder,'pre_processed_data','feature_engineering'),'tile_level_features/',name '/'];
    save_directory = output_folder;
    matlab_name=[save_directory,'gap_features_out.mat'];
    if exist(matlab_name) == 2
        load(matlab_name,'row_dim','col_dim','save_directory','ecm_mask','tissue_mask','discrete_gap_labels');
        quadrant_sf=4
        for tile_row = 0:quadrant_sf-1
            for tile_column = 0:quadrant_sf-1
                variable_names={};
                output_stats = [];
                gap_quadrant_analysis_scale_factor(variable_names,output_stats,quadrant_sf,tile_row,tile_column,row_dim,col_dim,save_directory,ecm_mask,tissue_mask,discrete_gap_labels);
            end
        end
        quadrant_sf=8
        for tile_row = 0:quadrant_sf-1
            for tile_column = 0:quadrant_sf-1
                variable_names={};
                output_stats = [];
                gap_quadrant_analysis_scale_factor(variable_names,output_stats,quadrant_sf,tile_row,tile_column,row_dim,col_dim,save_directory,ecm_mask,tissue_mask,discrete_gap_labels);
            end
        end

    end

%     else
%         'The ecm proportion is below threshold. No fibres can be extracted and there will be one large gap covering the whole tile.'
%     end

    
end
end

