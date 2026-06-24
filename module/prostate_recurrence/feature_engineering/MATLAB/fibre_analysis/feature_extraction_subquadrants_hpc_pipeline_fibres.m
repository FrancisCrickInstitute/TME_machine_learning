function feature_extraction_subquadrants_hpc_pipeline_fibres(image_start,image_end,image_folder,function_folder)

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
    matlab_name=[save_directory,'fibre_features_out.mat']
    if exist(matlab_name) == 2
        load(matlab_name,'row_dim','col_dim','save_directory','ecm_mask','tissue_mask','directory_ctfire_input','ctfire_file','image_folder','image_file','image','angle_continuum');
        csv_dir=[save_directory,'*','_features_out_quadrant_scalefactor_1_','*','.csv']; 
        csv_list = dir(csv_dir)
        whos
        img = eval('image');
        %img
        size(img);
        quadrant_sf=4
        for tile_row = 0:quadrant_sf-1
            for tile_column = 0:quadrant_sf-1
                variable_names={};
                output_stats = [];
                try
                    fibre_quadrant_analysis_scale_factor(variable_names,output_stats,quadrant_sf,tile_row,tile_column,row_dim,col_dim,save_directory,ecm_mask,tissue_mask,directory_ctfire_input,ctfire_file,image_folder,image_file,img,angle_continuum,csv_list);
                catch ME
                    'Error. Skipping quadrant.'
                end
            end
        end
        quadrant_sf=8
        for tile_row = 0:quadrant_sf-1
            for tile_column = 0:quadrant_sf-1
                variable_names={};
                output_stats = [];
                try
                    fibre_quadrant_analysis_scale_factor(variable_names,output_stats,quadrant_sf,tile_row,tile_column,row_dim,col_dim,save_directory,ecm_mask,tissue_mask,directory_ctfire_input,ctfire_file,image_folder,image_file,img,angle_continuum,csv_list);
                catch ME
                    'Error. Skipping quadrant.'
                end
            end
        end
        matlab_name=[save_directory,'gap_features_out.mat'];
        load(matlab_name,'row_dim','col_dim','save_directory','ecm_mask','tissue_mask','discrete_gap_labels');
        quadrant_sf=4
        for tile_row = 0:quadrant_sf-1
            for tile_column = 0:quadrant_sf-1
                variable_names={};
                output_stats = [];
                try
                    gap_quadrant_analysis_scale_factor(variable_names,output_stats,quadrant_sf,tile_row,tile_column,row_dim,col_dim,save_directory,ecm_mask,tissue_mask,discrete_gap_labels,csv_list);
                catch ME
                    'Error. Skipping quadrant.'
                end
            end
        end
        quadrant_sf=8
        for tile_row = 0:quadrant_sf-1
            for tile_column = 0:quadrant_sf-1
                variable_names={};
                output_stats = [];
                try
                    gap_quadrant_analysis_scale_factor(variable_names,output_stats,quadrant_sf,tile_row,tile_column,row_dim,col_dim,save_directory,ecm_mask,tissue_mask,discrete_gap_labels,csv_list);
                catch ME
                    'Error. Skipping quadrant.'
                end
            end
        end

    end

    
end
end




