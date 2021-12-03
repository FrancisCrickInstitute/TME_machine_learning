function feature_extraction_hpc_pipeline(image_start,image_end,image_folder,function_folder,ctfire_function_folder,radius_range,tissue_mask)
addpath(genpath(function_folder))
image_type = '.tif';
analysis_mode = '1';
input_image_dir = [image_folder '*' image_type] ;
input_image_list = dir(input_image_dir);
directory_ctfire_input=[image_folder,'ctFIREout/'];

current_dir = pwd;
output_filetype='tif';
for image_I=image_start:image_end
    variable_names={};
    output_stats = [];
    image_file = input_image_list(image_I).name;
    image = imread([image_folder '/' image_file]);
    if sum(tissue_mask(:))/length(tissue_mask(:)) >= 0.005%Threshold to accept tile
        %Run CT-Fire
        [~, name, ext] = fileparts([image_folder image_file]);
        image_folder
        output_folder = [strrep(image_folder,'pre_processed_data','feature_engineering'),'tile_level_features/',name '/'];
        str1 = strfind(output_folder,'/');
        str2 = strfind(output_folder,'feature_engineering/');
        str_index=str1(str1>str2);
        for str_i=1:length(str_index)
           directory_create=output_folder(1:str_index(str_i))
           mkdir(directory_create)
           fileattrib(directory_create,'+w','a') 
        end
        imageRange = num2str(image_I); % 'all': default, all images in the image folder ; or specified image indexes, such as '1:2', or '1:2 3'
        cd(ctfire_function_folder);
        CurveAlign_CommandLine(image_folder,image_type,analysis_mode,imageRange);
        cd([current_dir '/']);
        
        %Load CT-FIRE
        save_directory = [output_folder '/' name '_automated_results/'];
        save_directory = output_folder;
        mkdir(save_directory);
        fileattrib(save_directory,'+w','a');
        input_ctfire_dir = [directory_ctfire_input '*' name '*' '.mat'];
        input_ctfire_list = dir(input_ctfire_dir);
        ctfire_file=input_ctfire_list(1).name;
        [ctfire_fibres fibre_matrix discrete_fibres] = load_ctfire_data([directory_ctfire_input ctfire_file],[image_folder image_file],5);


        %Run Feature extraction
        
        %Plot discrete fibres
        fibre_matrix_zero=fibre_matrix;
        fibre_matrix_zero(isnan(fibre_matrix_zero))=0;
        save_filename = 'label_image_discrete';
        plotted_matrix = fibre_matrix;
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'parula',0,'fibre_label',[],[],tissue_mask);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %branchpoint, endpoint and shape analysis
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [fibre_skeleton,fibre_branchpoints,fibre_endpoints,fibre_disconnected_branches] = fibre_skeletonisation(fibre_matrix);
        
        %HDM normalised to mask area
        HDM=length(find(fibre_matrix>0))/sum(tissue_mask(:));
        variable_names={variable_names{1:end},'HDM'};
        output_stats = [output_stats,HDM];

        %Plot branchpoints
        branchpoint_image = fibre_matrix;
        branchpoint_image(fibre_matrix>0)=1;
        branchpoint_image(fibre_branchpoints>0)=2;
        branchpoint_image(fibre_endpoints>0)=3;
        save_filename = 'branch_image';
        plotted_matrix = branchpoint_image;
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'cool',0,'branch_point_label',[],[],tissue_mask);
        
        %Analyse fibre shapes
        [fibre_shapes] = fibre_shape(ctfire_fibres);

        branch_labels = bwlabel(fibre_branchpoints,8);
        endpoints_labels = bwlabel(fibre_endpoints,8);
        branch_pts_per_pixel = max(branch_labels(:))/sum([fibre_shapes.length]);
        end_pts_per_pixel = max(endpoints_labels(:))/sum([fibre_shapes.length]);
        variable_names={variable_names{1:end},'branch points','end points'};
        output_stats = [output_stats,branch_pts_per_pixel,end_pts_per_pixel];


        h=figure;
        histogram([fibre_shapes.length])
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Fibre length (pixels)','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory,'fibre_length_histogram.tif'];
        saveas(h,save_filename)
        close all

        h=figure;
        histogram([fibre_shapes.displacement])
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Fibre displacement (pixels)','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory,'fibre_displacement_histogram.tif'];
        saveas(h,save_filename)
        close all

        h=figure;
        histogram([fibre_shapes.persistence])
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Fibre persistence (pixels)','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory,'fibre_persistence_histogram.tif'];
        saveas(h,save_filename)
        close all
        
        h=figure;
        histogram([fibre_shapes.area],100)
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Fibre area (pixels^2)','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory,'fibre_area_histogram.tif'];
        saveas(h,save_filename)
        close all
        
        h=figure;
        histogram([fibre_shapes.perimeter])
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Fibre perimeter (pixels)','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory,'fibre_perimeter_histogram.tif'];
        saveas(h,save_filename)
        close all
        
        h=figure;
        histogram([fibre_shapes.circularity])
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Fibre circularity','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory,'fibre_circularity_histogram.tif'];
        saveas(h,save_filename)
        close all

        h=figure;
        histogram([fibre_shapes.length]./[fibre_shapes.perimeter],20)
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Fibre length/perimeter','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory,'fibre_length_over_perimeter_histogram.tif'];
        saveas(h,save_filename)
        close all

        Shape_stats(1,1)=mean([fibre_shapes.length]);
        Shape_stats(2,1)=std([fibre_shapes.length]);
        Shape_stats(3,1)=skewness([fibre_shapes.length]);
        Shape_stats(4,1)=kurtosis([fibre_shapes.length]);
        Shape_stats(1,2)=mean([fibre_shapes.displacement]);
        Shape_stats(2,2)=std([fibre_shapes.displacement]);
        Shape_stats(3,2)=skewness([fibre_shapes.displacement]);
        Shape_stats(4,2)=kurtosis([fibre_shapes.displacement]);
        Shape_stats(1,3)=mean([fibre_shapes.persistence]);
        Shape_stats(2,3)=std([fibre_shapes.persistence]);
        Shape_stats(3,3)=skewness([fibre_shapes.persistence]);
        Shape_stats(4,3)=kurtosis([fibre_shapes.persistence]);
        Shape_stats(1,4)=mean([fibre_shapes.area]);
        Shape_stats(2,4)=std([fibre_shapes.area]);
        Shape_stats(3,4)=skewness([fibre_shapes.area]);
        Shape_stats(4,4)=kurtosis([fibre_shapes.area]);
        Shape_stats(1,5)=mean([fibre_shapes.circularity]);
        Shape_stats(2,5)=std([fibre_shapes.circularity]);
        Shape_stats(3,5)=skewness([fibre_shapes.circularity]);
        Shape_stats(4,5)=kurtosis([fibre_shapes.circularity]);
        Shape_stats(1,6)=mean([fibre_shapes.length]./[fibre_shapes.perimeter]);
        Shape_stats(2,6)=std([fibre_shapes.length]./[fibre_shapes.perimeter]);
        Shape_stats(3,6)=skewness([fibre_shapes.length]./[fibre_shapes.perimeter]);
        Shape_stats(4,6)=kurtosis([fibre_shapes.length]./[fibre_shapes.perimeter]);

        variable_names={variable_names{1:end},'mean  fibre length','std  fibre length','skew  fibre length','kurtosis  fibre length','mean  fibre displacement','std  fibre displacement','skew  fibre displacement','kurtosis  fibre displacement',...
            'mean  fibre persistence','std  fibre persistence','skew  fibre persistence','kurtosis  fibre persistence','mean  fibre area','std  fibre area','skew  fibre area','kurtosis  fibre area',...
            'mean  fibre circularity','std  fibre circularity','skew  fibre circularity','kurtosis  fibre circularity',...
            'mean  fibre length/perim.','std  fibre length/perim.','skew  fibre length/perim.','kurtosis  fibre length/perim.'};
        output_stats = [output_stats,Shape_stats(1,1),Shape_stats(2,1),Shape_stats(3,1),Shape_stats(4,1),Shape_stats(1,2),Shape_stats(2,2),Shape_stats(3,2),Shape_stats(4,2),Shape_stats(1,3),Shape_stats(2,3),Shape_stats(3,3),Shape_stats(4,3),Shape_stats(1,4),Shape_stats(2,4),Shape_stats(3,4),Shape_stats(4,4),Shape_stats(1,5),Shape_stats(2,5),Shape_stats(3,5),Shape_stats(4,5),Shape_stats(1,6),Shape_stats(2,6),Shape_stats(3,6),Shape_stats(4,6)];



        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %discrete and continuum curvature analysis
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [discrete_fibre_curvature,curvature_matrix,curvature_continuum] = fibre_curvature(ctfire_fibres, discrete_fibres, fibre_matrix);
        save_filename = 'curvature_discrete';
        plotted_matrix = curvature_matrix; 
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',1,'Curvature',0.5,0,tissue_mask);
        save_filename = 'curvature_continuum';
        plotted_matrix = curvature_continuum;
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',1,'Curvature',0.5,0,tissue_mask);
        curvature=[];
        for I=1:length(discrete_fibre_curvature)
            if max(isnan(discrete_fibre_curvature(I).curvature))==0
                curvature=[curvature;discrete_fibre_curvature(I).curvature];
            end
        end

        %Plot histograms of curvature and calculate statistics
        h=figure;
        histogram(curvature)
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Curvature','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        max_xlim = max(curvature(~isoutlier(curvature)));
        order = floor(log10(max_xlim));
        max_xlim = ceil(max_xlim/10^(order-1))*10^(order-1);
        xlim([0,max_xlim])
        axis square;
        save_filename=[save_directory,'curvature_histogram.tif'];
        saveas(h,save_filename)
        close all
        mean_curvature=mean(curvature,'omitnan');
        std_curvature=std(curvature,'omitnan');
        skewness_curvature=skewness(curvature);
        kurtosis_curvature=kurtosis(curvature);
        prctile_75_curvature = prctile(curvature,75);
        prctile_90_curvature = prctile(curvature,90);
        prctile_95_curvature = prctile(curvature,95);
        prctile_99_curvature = prctile(curvature,99);
        variable_names={variable_names{1:end},'mean_curvature','std_curvature','skewness_curvature','kurtosis_curvature','prctile_75_curvature','prctile_90_curvature','prctile_95_curvature','prctile_99_curvature'};
        output_stats = [output_stats,mean_curvature,std_curvature,skewness_curvature,kurtosis_curvature,prctile_75_curvature,prctile_90_curvature,prctile_95_curvature,prctile_99_curvature];
        
        %Compare curvature distibution to Uniform and Gaussian
        %distributions
        x = (curvature-mean_curvature)/std_curvature;
        [h,p] = kstest(x);
        variable_names={variable_names{1:end},'curvature 1d Gaussian ks-test p-value'};
        output_stats = [output_stats,p];
        pd = makedist('Uniform','Lower',min(curvature,'omitnan'),'Upper',max(curvature,'omitnan'));
        [h,p] = kstest(x,'cdf',pd)
        variable_names={variable_names{1:end},'curvature 1d Uniform ks-test p-value'};
        output_stats = [output_stats,p];
        pd = makedist('Poisson','lambda',mean_curvature);
        [h,p] = kstest(curvature,'cdf',pd);
        variable_names={variable_names{1:end},'curvature 1d Poisson ks-test p-value'};
        output_stats = [output_stats,p];
        pd = makedist('Exponential','mu',mean_curvature);
        [h,p] = kstest(curvature,'cdf',pd);
        variable_names={variable_names{1:end},'curvature 1d Exponential ks-test p-value'};
        output_stats = [output_stats,p];



%         [h,p] = lillietest(curvature,'Distribution','exponential');
%         [h,p] = lillietest(curvature,'Distribution','normal');







        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %discrete and continuum angle analysis
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [discrete_fibre_angles,end_to_end_angle_matrix,end_to_end_x_derivative_matrix,end_to_end_y_derivative_matrix,local_angle_matrix,local_x_derivative_matrix,local_y_derivative_matrix] = fibre_angle(ctfire_fibres, discrete_fibres, fibre_matrix);
        [end_to_end_angle_continuum,end_to_end_x_derivative_continuum,end_to_end_y_derivative_continuum,local_angle_continuum,local_x_derivative_continuum,local_y_derivative_continuum] = fibre_angle_continuum(end_to_end_x_derivative_matrix,end_to_end_y_derivative_matrix,local_x_derivative_matrix,local_y_derivative_matrix);
        save_filename = 'local_angle_discrete';
        plotted_matrix = local_angle_matrix;
        periodic_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,1,tissue_mask);
        save_filename = 'endtoend_angle_discrete';
        plotted_matrix = end_to_end_angle_matrix;
        periodic_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,1,tissue_mask);
        save_filename = 'local_angle_continuum'; 
        plotted_matrix = local_angle_continuum;
        periodic_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,1,tissue_mask);
        save_filename = 'endtoend_angle_continuum'; 
        plotted_matrix = end_to_end_angle_continuum;
        periodic_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,1,tissue_mask);

        %Circular_stats
        local_angles=[];
        for I=1:length(discrete_fibre_angles)
            local_angles=[local_angles;discrete_fibre_angles(I).local_fibre_angle];
        end
        h=figure;
        histogram([local_angles])
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Angle (radians)','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory,'local_angles_histogram.tif'];
        saveas(h,save_filename)
        close all
        local_circ_stats=circ_stats(2*local_angles);
        end_to_end_angles=[];
        for I=1:length(discrete_fibre_angles)
            end_to_end_angles=[end_to_end_angles;discrete_fibre_angles(I).end_to_end_fibre_angle];
        end
        h=figure;
        histogram([end_to_end_angles])
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Angle (radians)','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory,'end_to_end_angles_histogram.tif'];
        saveas(h,save_filename)
        close all
        end_to_end_angles_circ_stats=circ_stats(2*end_to_end_angles);

        variable_names={variable_names{1:end},'local_angle_var','local_angle_std','local_angle_std0','local_angle_skew','local_angle_skew0','local_angle_kurtosis','local_angle_kurtosis0'};
        output_stats = [output_stats,local_circ_stats.var,local_circ_stats.std,local_circ_stats.std0,local_circ_stats.skewness,local_circ_stats.skewness0,local_circ_stats.kurtosis,local_circ_stats.kurtosis0];
        variable_names={variable_names{1:end},'end_to_end_angle_var','end_to_end_angle_std','end_to_end_angle_std0','end_to_end_angle_skew','end_to_end_angle_skew0','end_to_end_angle_kurtosis','end_to_end_angle_kurtosis0'};
        output_stats = [output_stats,end_to_end_circ_stats.var,end_to_end_circ_stats.std,end_to_end_circ_stats.std0,end_to_end_circ_stats.skewness,end_to_end_circ_stats.skewness0,end_to_end_circ_stats.kurtosis,end_to_end_circ_stats.kurtosis0];
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Convolution analysis
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        for R =1:length(radius_range)
            radius_input = radius_range(R);
            
            [angle_image_median_out,angle_length] = statistical_function_convolution(local_angle_continuum,radius_input,'median',1,1);
            save_filename_angle(R).median = ['angle_median_' num2str(radius_input)];
            angle_image_median(:,:,R)=angle_image_median_out;
                        
            [angle_image_std_out,angle_length] = statistical_function_convolution(local_angle_continuum,radius_input,'std',1,1);
            save_filename_angle(R).std = ['angle_std_' num2str(radius_input)];
            angle_image_std(:,:,R)=angle_image_std_out;
            

            [angle_image_lq_out,angle_length] = statistical_function_convolution(local_angle_continuum,radius_input,'lower quartile',1,1);
            save_filename_angle(R).lq = ['angle_lq_' num2str(radius_input)];
            angle_image_lq(:,:,R)=angle_image_lq_out;

            [angle_image_uq_out,angle_length] = statistical_function_convolution(local_angle_continuum,radius_input,'upper quartile',1,1);
            save_filename_angle(R).uq = ['angle_uq_' num2str(radius_input)];
            angle_image_uq(:,:,R)=angle_image_uq_out;

            [angle_image_skew_out,angle_length] = statistical_function_convolution(local_angle_continuum,radius_input,'skewness',1,1);
            save_filename_angle(R).skew = ['skew_' num2str(radius_input)];
            angle_image_skew(:,:,R)=angle_image_skew_out;

            [angle_image_kurt_out,angle_length] = statistical_function_convolution(local_angle_continuum,radius_input,'kurtosis',1,1);
            save_filename_angle(R).kurt = ['kurt_' num2str(radius_input)];
            angle_image_kurt(:,:,R)=angle_image_kurt_out;

            [curvature_image_median_out,curvature_length] = statistical_function_convolution(curvature_continuum,radius_input,'median',1,2);
            save_filename_curvature(R).median = ['curvature_median_' num2str(radius_input)];
            curvature_image_median(:,:,R)=curvature_image_median_out;
            
            [curvature_image_std_out,curvature_length] = statistical_function_convolution(curvature_continuum,radius_input,'std',1,2);
            save_filename_curvature(R).std = ['curvature_std_' num2str(radius_input)];
            curvature_image_std(:,:,R)=curvature_image_std_out;

            [curvature_image_lq_out,curvature_length] = statistical_function_convolution(curvature_continuum,radius_input,'lower quartile',1,2);
            save_filename_curvature(R).lq = ['curvature_lq_' num2str(radius_input)];
            curvature_image_lq(:,:,R)=curvature_image_lq_out;

            [curvature_image_uq_out,curvature_length] = statistical_function_convolution(curvature_continuum,radius_input,'upper quartile',1,2);
            save_filename_curvature(R).uq = ['curvature_uq_' num2str(radius_input)];
            curvature_image_uq(:,:,R)=curvature_image_uq_out;

            [curvature_image_skew_out,curvature_length] = statistical_function_convolution(curvature_continuum,radius_input,'skewness',1,2);
            save_filename_curvature(R).skew = ['skew_' num2str(radius_input)];
            curvature_image_skew(:,:,R)=curvature_image_skew_out;

            [curvature_image_kurt_out,curvature_length] = statistical_function_convolution(curvature_continuum,radius_input,'kurtosis',1,2);
            save_filename_curvature(R).kurt = ['kurt_' num2str(radius_input)];
            curvature_image_kurt(:,:,R)=curvature_image_kurt_out;
            
            mean_angle_stat_conv(R,1)=mean(angle_image_median_out.*tissue_mask,'all','omitnan');
            mean_angle_stat_conv(R,2)=mean(angle_image_std_out.*tissue_mask,'all','omitnan');
            mean_angle_stat_conv(R,3)=mean(angle_image_lq_out.*tissue_mask,'all','omitnan');
            mean_angle_stat_conv(R,4)=mean(angle_image_uq_out.*tissue_mask,'all','omitnan');
            mean_angle_stat_conv(R,5)=mean(angle_image_skew_out.*tissue_mask,'all','omitnan');
            mean_angle_stat_conv(R,6)=mean(angle_image_kurt_out.*tissue_mask,'all','omitnan');

            mean_curvature_stat_conv(R,1)=mean(curvature_image_median_out.*tissue_mask,'all','omitnan');
            mean_curvature_stat_conv(R,2)=mean(curvature_image_std_out.*tissue_mask,'all','omitnan');
            mean_curvature_stat_conv(R,3)=mean(curvature_image_lq_out.*tissue_mask,'all','omitnan');
            mean_curvature_stat_conv(R,4)=mean(curvature_image_uq_out.*tissue_mask,'all','omitnan');
            mean_curvature_stat_conv(R,5)=mean(curvature_image_skew_out.*tissue_mask,'all','omitnan');
            mean_curvature_stat_conv(R,6)=mean(curvature_image_kurt_out.*tissue_mask,'all','omitnan');
            variable_names={variable_names{1:end},['mean_median_angle_rad_' num2str(radius_range(R)) '_conv']};
            output_stats = [output_stats,mean_angle_stat_conv(R,1)];
            variable_names={variable_names{1:end},['mean_std_angle_rad_' num2str(radius_range(R)) '_conv']};
            output_stats = [output_stats,mean_angle_stat_conv(R,2)];
            variable_names={variable_names{1:end},['mean_lq_angle_rad_' num2str(radius_range(R)) '_conv']};
            output_stats = [output_stats,mean_angle_stat_conv(R,3)];
            variable_names={variable_names{1:end},['mean_uq_angle_rad_' num2str(radius_range(R)) '_conv']};
            output_stats = [output_stats,mean_angle_stat_conv(R,4)];
            variable_names={variable_names{1:end},['mean_skew_angle_rad_' num2str(radius_range(R)) '_conv']};
            output_stats = [output_stats,mean_angle_stat_conv(R,5)];
            variable_names={variable_names{1:end},['mean_kurt_angle_rad_' num2str(radius_range(R)) '_conv']};
            output_stats = [output_stats,mean_angle_stat_conv(R,6)];
            variable_names={variable_names{1:end},['mean_median_curvature_rad_' num2str(radius_range(R)) '_conv']};
            output_stats = [output_stats,mean_curvature_stat_conv(R,1)];
            variable_names={variable_names{1:end},['mean_std_curvature_rad_' num2str(radius_range(R)) '_conv']};
            output_stats = [output_stats,mean_curvature_stat_conv(R,2)];
            variable_names={variable_names{1:end},['mean_lq_curvature_rad_' num2str(radius_range(R)) '_conv']};
            output_stats = [output_stats,mean_curvature_stat_conv(R,3)];
            variable_names={variable_names{1:end},['mean_uq_curvature_rad_' num2str(radius_range(R)) '_conv']};
            output_stats = [output_stats,mean_curvature_stat_conv(R,4)];
            variable_names={variable_names{1:end},['mean_skew_curvature_rad_' num2str(radius_range(R)) '_conv']};
            output_stats = [output_stats,mean_curvature_stat_conv(R,5)];
            variable_names={variable_names{1:end},['mean_kurt_curvature_rad_' num2str(radius_range(R)) '_conv']};
            output_stats = [output_stats,mean_curvature_stat_conv(R,6)];
        end

        max_val_angle_median=max(angle_image_median(:),'omitnan');
        min_val_angle_median = 0;
        colormap_angle_median='median_angle_difference_heatmap';
        max_val_angle_std=max(angle_image_std(:),'omitnan');
        min_val_angle_std = min(angle_image_std(:),'omitnan');
        colormap_angle_std='std_angle_difference_heatmap';
        max_val_angle_lq=max(angle_image_lq(:),'omitnan');
        min_val_angle_lq = min(angle_image_lq(:),'omitnan');
        colormap_angle_lq='lq_angle_difference_heatmap';
        max_val_angle_uq=max(angle_image_uq(:),'omitnan');
        min_val_angle_uq = min(angle_image_uq(:),'omitnan');
        colormap_angle_uq='uq_angle_difference_heatmap';
        max_val_angle_skew=max(angle_image_skew(:),'omitnan');
        min_val_angle_skew = min(angle_image_skew(:),'omitnan');
        colormap_angle_skew='skew_angle_difference_heatmap';
        max_val_angle_kurt=max(angle_image_kurt(:),'omitnan');
        min_val_angle_kurt = min(angle_image_kurt(:),'omitnan');
        colormap_angle_kurt='kurt_angle_difference_heatmap';

        max_val_curvature_median=max(curvature_image_median(:),'omitnan');
        min_val_curvature_median = 0;
        colormap_curvature_median='median_curvature_difference_heatmap';
        max_val_curvature_std=max(curvature_image_std(:),'omitnan');
        min_val_curvature_std = min(curvature_image_std(:),'omitnan');
        colormap_curvature_std='std_curvature_difference_heatmap';
        max_val_curvature_lq=max(curvature_image_lq(:),'omitnan');
        min_val_curvature_lq = min(curvature_image_lq(:),'omitnan');
        colormap_curvature_lq='lq_curvature_difference_heatmap';
        max_val_curvature_uq=max(curvature_image_uq(:),'omitnan');
        min_val_curvature_uq = min(curvature_image_uq(:),'omitnan');
        colormap_curvature_uq='uq_curvature_difference_heatmap';
        max_val_curvature_skew=max(curvature_image_skew(:),'omitnan');
        min_val_curvature_skew = min(curvature_image_skew(:),'omitnan');
        colormap_curvature_skew='skew_curvature_difference_heatmap';
        max_val_curvature_kurt=max(curvature_image_kurt(:),'omitnan');
        min_val_curvature_kurt = min(curvature_image_kurt(:),'omitnan');
        colormap_curvature_kurt='kurt_curvature_difference_heatmap';

        for R=1:length(radius_range)
            linear_colourmap(save_directory,save_filename_angle(R).median,output_filetype,angle_image_median(:,:,R),'parula',1,colormap_angle_median,max_val_angle_median,min_val_angle_median,tissue_mask);
            linear_colourmap(save_directory,save_filename_angle(R).std,output_filetype,angle_image_std(:,:,R),'parula',1,colormap_angle_std,max_val_angle_std,min_val_angle_std,tissue_mask);
            linear_colourmap(save_directory,save_filename_angle(R).lq,output_filetype,angle_image_lq(:,:,R),'parula',1,colormap_angle_lq,max_val_angle_lq,min_val_angle_lq,tissue_mask);
            linear_colourmap(save_directory,save_filename_angle(R).uq,output_filetype,angle_image_uq(:,:,R),'parula',1,colormap_angle_uq,max_val_angle_uq,min_val_angle_uq,tissue_mask);
            linear_colourmap(save_directory,save_filename_angle(R).skew,output_filetype,angle_image_skew(:,:,R),'parula',1,colormap_angle_skew,max_val_angle_skew,min_val_angle_skew,tissue_mask);
            linear_colourmap(save_directory,save_filename_angle(R).kurt,output_filetype,angle_image_kurt(:,:,R),'parula',1,colormap_angle_kurt,max_val_angle_kurt,min_val_angle_kurt,tissue_mask);

            linear_colourmap(save_directory,save_filename_curvature(R).median,output_filetype,curvature_image_median(:,:,R),'parula',1,colormap_curvature_median,max_val_curvature_median,min_val_curvature_median,tissue_mask);
            linear_colourmap(save_directory,save_filename_curvature(R).std,output_filetype,curvature_image_std(:,:,R),'parula',1,colormap_curvature_std,max_val_curvature_std,min_val_curvature_std,tissue_mask);
            linear_colourmap(save_directory,save_filename_curvature(R).lq,output_filetype,curvature_image_lq(:,:,R),'parula',1,colormap_curvature_lq,max_val_curvature_lq,min_val_curvature_lq,tissue_mask);
            linear_colourmap(save_directory,save_filename_curvature(R).uq,output_filetype,curvature_image_uq(:,:,R),'parula',1,colormap_curvature_uq,max_val_curvature_uq,min_val_curvature_uq,tissue_mask);
            linear_colourmap(save_directory,save_filename_curvature(R).skew,output_filetype,curvature_image_skew(:,:,R),'parula',1,colormap_curvature_skew,max_val_curvature_skew,min_val_curvature_skew,tissue_mask);
            linear_colourmap(save_directory,save_filename_curvature(R).kurt,output_filetype,curvature_image_kurt(:,:,R),'parula',1,colormap_curvature_kurt,max_val_curvature_kurt,min_val_curvature_kurt,tissue_mask);
       end
            
%         %lq,uq,skewness,kurtosis
%         linear_colourmap(save_directory,save_filename10,output_filetype,input10,'parula',1,colormap_name,max_val,0);
%         linear_colourmap(save_directory,save_filename10,output_filetype,input10,'parula',1,colormap_name,max_val,0);
%         linear_colourmap(save_directory,save_filename10,output_filetype,input10,'parula',1,colormap_name,4,-2);
%         linear_colourmap(save_directory,save_filename10,output_filetype,input10,'parula',1,colormap_name,10,0);

%         %curvature e.g. ranges median, std, lq, uq, skew, kurt
%         linear_colourmap(save_directory,save_filename10,output_filetype,input10,'parula',1,colormap_name,0.3,0);
%         linear_colourmap(save_directory,save_filename10,output_filetype,input10,'parula',1,colormap_name,0.5,0);
%         linear_colourmap(save_directory,save_filename10,output_filetype,input10,'parula',1,colormap_name,0.50,0);
%         linear_colourmap(save_directory,save_filename10,output_filetype,input10,'parula',1,colormap_name,0.5,0);
%         linear_colourmap(save_directory,save_filename10,output_filetype,input10,'parula',1,colormap_name,15,0);
%         linear_colourmap(save_directory,save_filename10,output_filetype,input10,'parula',1,colormap_name,100,0);
        


        h=figure;
        plot(radius_range,mean_angle_stat_conv(:,1)/mean_angle_stat_conv(1,1),'LineWidth',3)
        hold on
        for s=2:6
            plot(radius_range,mean_angle_stat_conv(:,s)/mean_angle_stat_conv(1,s),'LineWidth',3);
        end
        legend('Median','Stan. dev.','Lower quart.','Upper quart.','Skewness','Kurtosis','location','northeastoutside')
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Radius (pixels)','fontsize',24,'fontweight','b')
        ylabel('Normalised mean value','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory,'angle_convolution_lineplots.tif'];
        saveas(h,save_filename)
        close all

        
        h=figure;
        plot(radius_range,mean_curvature_stat_conv(:,1)/mean_curvature_stat_conv(1,1),'LineWidth',3)
        hold on
        for s=2:6
            plot(radius_range,mean_curvature_stat_conv(:,s)/mean_curvature_stat_conv(1,s),'LineWidth',3);
        end
        legend('Median','Stan. dev.','Lower quart.','Upper quart.','Skewness','Kurtosis','location','northeastoutside')
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Radius (pixels)','fontsize',24,'fontweight','b')
        ylabel('Normalised mean value','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory,'curvature_convolution_lineplots.tif']
        saveas(h,save_filename)
        close all


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %branchpoint, endpoint and branches overlay on continuum fields
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        
        %Overlay branchpoints and endpoints over curvature
        branches_curvature = curvature_continuum;
        branches_curvature(fibre_branchpoints>0) = NaN;
        save_filename = 'branchpoints_curvature_continuum';
        plotted_matrix = branches_curvature;
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',1,'Curvature',0.5,0,tissue_mask);

        endpoints_curvature = curvature_continuum;
        endpoints_curvature(fibre_endpoints>0) = NaN;
        save_filename = 'endpoints_curvature_continuum'; 
        plotted_matrix = endpoints_curvature; 
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',0,'Curvature',0.5,0,tissue_mask);

        branches_curvature = curvature_continuum;
        branches_curvature(fibre_disconnected_branches>0) = NaN;
        save_filename = 'branches_curvature_continuum';
        plotted_matrix = branches_curvature;
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',0,'Curvature',0.5,0,tissue_mask);

        %Curvature versus branch/end points scatter
        cm=winter(3)
        h=figure;
        scatter(zeros(length(curvature_continuum(fibre_branchpoints>0)),1)+1,curvature_continuum(fibre_branchpoints>0),10,cm(1,:),'filled','jitter','on')
        hold on
        scatter(zeros(length(curvature_continuum(fibre_endpoints>0)),1)+2,curvature_continuum(fibre_endpoints>0),10,cm(2,:),'filled','jitter','on')
        scatter(zeros(length(curvature_continuum(fibre_disconnected_branches>0)),1)+3,curvature_continuum(fibre_disconnected_branches>0),10,cm(3,:),'filled','jitter','on')
        plot([0.75,1.50],[median(curvature_continuum(fibre_branchpoints>0)),median(curvature_continuum(fibre_branchpoints>0))],'k','LineWidth',3)
        plot([0.8,1.2],[prctile(curvature_continuum(fibre_branchpoints>0),50),prctile(curvature_continuum(fibre_branchpoints>0),50)],'k','LineWidth',3)
        plot([0.8,1.2],[prctile(curvature_continuum(fibre_branchpoints>0),75),prctile(curvature_continuum(fibre_branchpoints>0),75)],'k','LineWidth',3)
        plot([1,1],[prctile(curvature_continuum(fibre_branchpoints>0),50),prctile(curvature_continuum(fibre_branchpoints>0),75)],'k','LineWidth',3)
        plot([1.75,2.50],[median(curvature_continuum(fibre_endpoints>0)),median(curvature_continuum(fibre_endpoints>0))],'k','LineWidth',3)
        plot([1.8,2.2],[prctile(curvature_continuum(fibre_endpoints>0),50),prctile(curvature_continuum(fibre_endpoints>0),50)],'k','LineWidth',3)
        plot([1.8,2.2],[prctile(curvature_continuum(fibre_endpoints>0),75),prctile(curvature_continuum(fibre_endpoints>0),75)],'k','LineWidth',3)
        plot([2,2],[prctile(curvature_continuum(fibre_endpoints>0),50),prctile(curvature_continuum(fibre_endpoints>0),75)],'k','LineWidth',3)
        plot([2.75,3.50],[median(curvature_continuum(fibre_disconnected_branches>0)),median(curvature_continuum(fibre_disconnected_branches>0))],'k','LineWidth',3)
        plot([2.8,3.2],[prctile(curvature_continuum(fibre_disconnected_branches>0),50),prctile(curvature_continuum(fibre_disconnected_branches>0),50)],'k','LineWidth',3)
        plot([2.8,3.2],[prctile(curvature_continuum(fibre_disconnected_branches>0),75),prctile(curvature_continuum(fibre_disconnected_branches>0),75)],'k','LineWidth',3)
        plot([3,3],[prctile(curvature_continuum(fibre_disconnected_branches>0),50),prctile(curvature_continuum(fibre_disconnected_branches>0),75)],'k','LineWidth',3)
        xlim([0.5,3.5])
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Branch type','fontsize',24,'fontweight','b')
        ylabel('Curvature','fontsize',24,'fontweight','b')
        xticks(1:3)
        xticklabels({'Branch point','End point','Branch'})
        axis square;
        save_filename=[save_directory,'curvature_differences_branches.tif'];
        saveas(h,save_filename)
        close all

        variable_names={variable_names{1:end},'ttest curvature branchpts vs endpoints'};
        [h,p]=ttest2(curvature_continuum(fibre_branchpoints>0),curvature_continuum(fibre_endpoints>0))
        output_stats = [output_stats,p];
        variable_names={variable_names{1:end},'ttest curvature branchpts vs branches'};
        [h,p]=ttest2(curvature_continuum(fibre_branchpoints>0),curvature_continuum(fibre_disconnected_branches>0))
        output_stats = [output_stats,p];
        variable_names={variable_names{1:end},'ttest curvature branches vs endpoints'};
        [h,p]=ttest2(curvature_continuum(fibre_endpoints>0),curvature_continuum(fibre_disconnected_branches>0))
        output_stats = [output_stats,p];
        g1=ones(1,size(curvature_continuum(fibre_branchpoints>0),1));
        g2=2*ones(1,size(curvature_continuum(fibre_endpoints>0),1));
        g3=3*ones(1,size(curvature_continuum(fibre_disconnected_branches>0),1));
        x=[curvature_continuum(fibre_branchpoints>0);curvature_continuum(fibre_endpoints>0);curvature_continuum(fibre_disconnected_branches>0)]' ;
        g=[g1,g2,g3]; 
        p=anova1(x,g,'off');
        variable_names={variable_names{1:end},['anova curvature branches vs endpoints vs branches']};
        output_stats = [output_stats,p];        
        
        


        cm=winter(3)
        for R=1:length(radius_range)
            h=figure;
            input = angle_image_median(:,:,R);
            scatter(zeros(length(input(fibre_branchpoints>0)),1)+R,input(fibre_branchpoints>0),10,cm(1,:),'filled','jitter','on')
            hold on
            scatter(zeros(length(input(fibre_endpoints>0)),1)+2,input(fibre_endpoints>0),10,cm(2,:),'filled','jitter','on')
            scatter(zeros(length(input(fibre_disconnected_branches>0)),1)+3,input(fibre_disconnected_branches>0),10,cm(3,:),'filled','jitter','on')
            plot([0.75,1.50],[median(input(fibre_branchpoints>0)),median(input(fibre_branchpoints>0))],'k','LineWidth',3)
            plot([0.8,1.2],[prctile(input(fibre_branchpoints>0),50),prctile(input(fibre_branchpoints>0),50)],'k','LineWidth',3)
            plot([0.8,1.2],[prctile(input(fibre_branchpoints>0),75),prctile(input(fibre_branchpoints>0),75)],'k','LineWidth',3)
            plot([1,1],[prctile(input(fibre_branchpoints>0),50),prctile(input(fibre_branchpoints>0),75)],'k','LineWidth',3)
            plot([1.75,2.50],[median(input(fibre_endpoints>0)),median(input(fibre_endpoints>0))],'k','LineWidth',3)
            plot([1.8,2.2],[prctile(input(fibre_endpoints>0),50),prctile(input(fibre_endpoints>0),50)],'k','LineWidth',3)
            plot([1.8,2.2],[prctile(input(fibre_endpoints>0),75),prctile(input(fibre_endpoints>0),75)],'k','LineWidth',3)
            plot([2,2],[prctile(input(fibre_endpoints>0),50),prctile(input(fibre_endpoints>0),75)],'k','LineWidth',3)
            plot([2.75,3.50],[median(input(fibre_disconnected_branches>0)),median(input(fibre_disconnected_branches>0))],'k','LineWidth',3)
            plot([2.8,3.2],[prctile(input(fibre_disconnected_branches>0),50),prctile(input(fibre_disconnected_branches>0),50)],'k','LineWidth',3)
            plot([2.8,3.2],[prctile(input(fibre_disconnected_branches>0),75),prctile(input(fibre_disconnected_branches>0),75)],'k','LineWidth',3)
            plot([3,3],[prctile(input(fibre_disconnected_branches>0),50),prctile(input(fibre_disconnected_branches>0),75)],'k','LineWidth',3)
            xlim([0.5,3.5])
            set(gca,'LineWidth',4.5)
            set(gca,'FontSize',20);
            xlabel('Branch type','fontsize',24,'fontweight','b')
            ylabel(['Median angle difference (rad. ' num2str(radius_range(R)) ')'],'fontsize',24,'fontweight','b')
            xticks(1:3)
            xticklabels({'Branch point','End point','Branch'})
            axis square;
            save_filename=[save_directory,'angle' num2str(radius_range(R)) '_differences_branches.tif'];
            saveas(h,save_filename)
            close all
            [h,p]=ttest2(input(fibre_branchpoints>0),input(fibre_endpoints>0));
            variable_names={variable_names{1:end},['ttest angle ' num2str(radius_range(R)) ' branchpts vs endpoints']};
            output_stats = [output_stats,p];
            [h,p]=ttest2(input(fibre_branchpoints>0),input(fibre_disconnected_branches>0));
            variable_names={variable_names{1:end},['ttest angle ' num2str(radius_range(R)) ' branchpts vs branches']};
            output_stats = [output_stats,p];
            [h,p]=ttest2(input(fibre_endpoints>0),input(fibre_disconnected_branches>0));
            variable_names={variable_names{1:end},['ttest angle ' num2str(radius_range(R)) ' branches vs endpoints']};
            output_stats = [output_stats,p];

            g1=ones(1,size(input(fibre_branchpoints>0),1));
            g2=2*ones(1,size(input(fibre_endpoints>0),1));
            g3=3*ones(1,size(input(fibre_disconnected_branches>0),1));
            x=[input(fibre_branchpoints>0);input(fibre_endpoints>0);input(fibre_disconnected_branches>0)]' ;
            g=[g1,g2,g3]; 
            p=anova1(x,g,'off');
            variable_names={variable_names{1:end},['anova angle ' num2str(radius_range(R)) ' branches vs endpoints vs branches']};
            output_stats = [output_stats,p];


        end

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Spatial derivatives
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [curvature_X,curvature_Y] = gradient(curvature_continuum); 
        save_filename = 'curvature_X_derivative';
        max_grad=max(max(curvature_X(:)),max(curvature_Y(:)))
        min_grad=min(min(curvature_X(:)),min(curvature_Y(:)))
        plotted_matrix = curvature_X; 
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',1,'Curvature_x_derivative_cm',0.05,-0.05,tissue_mask);
        save_filename = 'curvature_Y_derivative'; 
        plotted_matrix = curvature_Y; 
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',1,'Curvature_x_derivative_cm',0.05,-0.05,tissue_mask);
        curvature_X = curvature_X(tissue_mask);
        curvature_Y = curvature_Y(tissue_mask);
        curvature_derivative_correlation = corr(curvature_X(:),curvature_Y(:));

        variable_names={variable_names{1:end},'curvature_derivative_correlation'};
        output_stats = [output_stats,curvature_derivative_correlation];


        for R=1:length(radius_range)
            [angle_median_X,angle_median_Y] = gradient(angle_image_median(:,:,R).*tissue_mask);
            save_filename = ['angle_median' num2str(radius_range(R)) '_X_derivative'];
            max_grad=max(max(angle_median_X(:)),max(angle_median_Y(:)));
            min_grad=min(min(angle_median_X(:)),min(angle_median_Y(:)));
            plotted_matrix = angle_median_X;
            linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',1,'angle_median10_x_derivative_cm',max_grad,min_grad,tissue_mask);
            save_filename = ['angle_median' num2str(radius_range(R)) '_Y_derivative'];
            plotted_matrix = angle_median_Y; 
            linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',1,'angle_median1_x_derivative_cm',max_grad,min_grad,tissue_mask);
            angle_median_X = angle_median_X(tissue_mask);
            angle_median_X = angle_median_Y(tissue_mask);
            angle_derivative_correlation(R) = corr(angle_median_X(:),angle_median_Y(:));

            variable_names={variable_names{1:end},['angle_' num2str(radius_range(R)) '_derivative_correlation']};
            output_stats = [output_stats,angle_derivative_correlation(R)];

        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Randomness of spatial distribution of curvature and alignment
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %Are distributions of curvature etc. uniformly spatially distributed?
        %Looking at uniform distribution  
        Ufm=tissue_mask;
        Ufm=Ufm./sum(Ufm(:));
        [rows,cols]=size(fibre_matrix);
        [x,y] = meshgrid(1:cols,1:rows);
        curvature_dis=curvature_continuum.*tissue_mask;
        curvature_dis=curvature_dis./sum(curvature_dis(:));
        h=figure;
        surf(x,y,curvature_dis, 'edgecolor','none')
        axis;
        set(gca,'LineWidth',3.5)
        set(gca,'FontSize',20);
        xlabel('x','fontsize',24,'fontweight','b')
        ylabel('y','fontsize',24,'fontweight','b')
        zlabel('Density','fontsize',24,'fontweight','b')
        view(55,55)
        ImName=[save_directory 'density_curvature.tif']
        print(h,ImName, '-dtiff'); 
        close(h);
        h=figure;
        surf(x,y,Ufm, 'edgecolor','none')
        axis;
        set(gca,'LineWidth',3.5)
        set(gca,'FontSize',20);
        xlabel('x','fontsize',24,'fontweight','b')
        ylabel('y','fontsize',24,'fontweight','b')
        zlabel('Density','fontsize',24,'fontweight','b')
        view(55,55)
        ImName=[save_directory 'density_uniform.tif']
        print(h,ImName, '-dtiff'); 
        close(h);
        Z1=curvature_dis(:);
        Z2=Ufm(:);
        M1=(Z1+Z2)./2;
        for J=1:length(M1)
            if(Z1(J)~=0)
               ys1(J)=Z1(J)*log2(Z1(J)/M1(J)) ;
            else
               ys1(J)=0;
            end
            if(Z2(J)~=0)
               ys2(J)=Z2(J)*log2(Z2(J)/M1(J)) ;
            else
               ys2(J)=0;
            end
        end
        jensen_shannon=(0.5*sum(ys1)+0.5*sum(ys2))^0.5
        bhattacharyya=sum((Z1(:).*Z2(:)).^0.5);
        hellinger=(sum((Z1.^0.5-Z2.^0.5).^2))^0.5/(2^0.5);
        distribution_distance(1,1)=jensen_shannon;
        distribution_distance(2,1)=bhattacharyya;
        distribution_distance(3,1)=hellinger;
        variable_names={variable_names{1:end},'Curvature jensen shannon', 'Curvature Bhattacharyya', 'Curvature Hellinger'};
        output_stats = [output_stats,jensen_shannon,bhattacharyya,hellinger];




        for R=1:length(radius_range)
            angle_dis(:,:) = angle_image_median(:,:,R).*tissue_mask;
            angle_dis=angle_dis./sum(angle_dis(:));
            h=figure;
            surf(x,y,angle_dis, 'edgecolor','none')
            axis;
            set(gca,'LineWidth',3.5)
            set(gca,'FontSize',20);
            xlabel('x','fontsize',24,'fontweight','b')
            ylabel('y','fontsize',24,'fontweight','b')
            zlabel('Density','fontsize',24,'fontweight','b')
            view(55,55)
            ImName=[save_directory 'density_angle_dis' num2str(radius_range(R)) '.tif']
            print(h,ImName, '-dtiff'); 
            close(h);
    
            Z1=angle_dis(:);
            Z2=Ufm(:);
            M1=(Z1+Z2)./2;
            for J=1:length(M1)
                if(Z1(J)~=0)
                   ys1(J)=Z1(J)*log2(Z1(J)/M1(J)) ;
                else
                   ys1(J)=0;
                end
                if(Z2(J)~=0)
                   ys2(J)=Z2(J)*log2(Z2(J)/M1(J)) ;
                else
                   ys2(J)=0;
                end
            end
            jensen_shannon=(0.5*sum(ys1)+0.5*sum(ys2))^0.5
            bhattacharyya=sum((Z1(:).*Z2(:)).^0.5);
            hellinger=(sum((Z1.^0.5-Z2.^0.5).^2))^0.5/(2^0.5);
            distribution_distance(1,R+1)=jensen_shannon;
            distribution_distance(2,R+1)=bhattacharyya;
            distribution_distance(3,R+1)=hellinger;
    
            variable_names={variable_names{1:end},['Angle ' num2str(radius_range(R)) ' jensen shannon'], ['Angle ' num2str(radius_range(R)) ' Bhattacharyya'], ['Angle ' num2str(radius_range(R)) ' Hellinger']};
            output_stats = [output_stats,jensen_shannon,bhattacharyya,hellinger];

            angle_1d(:,:) = angle_image_median(:,:,R);
            angle_1d = angle_1d(tissue_mask);
            x = (angle_1d-mean(angle_1d(:),'omitnan'))/std(angle_1d(:),'omitnan');
            [h,p] = kstest(x);
            variable_names={variable_names{1:end},['median alignment radius ' num2str(radius_range(R)) ' 1d Gaussian ks-test p-value']};
            output_stats = [output_stats,p];
            pd = makedist('Uniform','Lower',min(angle_1d(:),'omitnan'),'Upper',max(angle_1d(:),'omitnan'));
            [h,p] = kstest(x,'cdf',pd)
            variable_names={variable_names{1:end},['median alignment radius ' num2str(radius_range(R)) ' 1d Uniform ks-test p-value']};
            output_stats = [output_stats,p];
            pd = makedist('Poisson','lambda',mean(angle_1d(:),'omitnan'));
            [h,p] = kstest(curvature,'cdf',pd);
            variable_names={variable_names{1:end},['median alignment radius ' num2str(radius_range(R)) ' 1d Poisson ks-test p-value']};
            output_stats = [output_stats,p];
            pd = makedist('Exponential','mu',mean(angle_1d(:),'omitnan'));
            [h,p] = kstest(curvature,'cdf',pd);
            variable_names={variable_names{1:end},['median alignment radius ' num2str(radius_range(R)) ' 1d Exponential ks-test p-value']};
            output_stats = [output_stats,p];
    
        end

        %Fractal dimension
        fractal_range=[1,4,16,64,256];
        [n,r] = boxcount(input_gap_matrix,'slope');
        x=log(r);
        y=log(n);
        dy=-gradient(y)./gradient(x);
        mdl = fitlm(r(1:end),dy(1:end),'constant','RobustOpts','on')
        coeffs = mdl.Coefficients.Estimate;
        variable_names={variable_names{1:end},['fractal dimension whole']};
        output_stats = [output_stats,coeffs(1)];

        h=figure;
        loglog(r,n, 'k', 'lineWidth',3)
        axis;
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('${\varepsilon}$, box size (pixels)','interpreter','latex','fontsize',24,'fontweight','b')
        ylabel('${N(\varepsilon)}$, number of boxes','interpreter','latex','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory 'fractal_loglog_plot.tif']
        saveas(h,save_filename)
        close all


        h=figure;
        semilogx(x, dy, 'k', 'lineWidth',3)
        axis;
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('${\varepsilon}$, box size (pixels)','interpreter','latex','fontsize',24,'fontweight','b')
        ylabel('${-d\ln(N(\varepsilon))/d\ln(\varepsilon)}$','interpreter','latex','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory 'fractal_semilog_derivative_plot.tif']
        saveas(h,save_filename)
        close all
        


        for I=1:length(fractal_range)-1
            lb=find(r==fractal_range(I));
            ub=find(r==fractal_range(I+1));
            mdl = fitlm(r(lb:ub),dy(lb:ub),'constant','RobustOpts','on')
            coeffs = mdl.Coefficients.Estimate;
            variable_names={variable_names{1:end},['fractal dimension range ' num2str(r(lb)) ' to ' num2str(r(ub))]};
            output_stats = [output_stats,coeffs(1)];

        end



        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Gap Analysis
        %Needs updating to include second mask
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        
        input_gap_matrix = fibre_matrix; 

        [...
          label_matrix,...
          radius_label_matrix,...
          centroid_row,...
          centroid_col,...
          circle_radius...
        ] = circle_gap_fitting(input_gap_matrix,0);
        ImName=[save_directory 'overlaid_gaps.tif']
        circle_gap_plotting(label_matrix,ImName);
        
        radii_vector = [5,10];
        [...
          area_weighted_sample,...
          neighbours,...
          variable_radius ...
          ] = circle_gap_statistics(label_matrix,radius_label_matrix,centroid_row,centroid_col,circle_radius,input_gap_matrix,radii_vector)



        output_stats = [output_stats,length(circle_radius)];
        variable_names={variable_names{1:end},['Total circles']};
        output_stats = [output_stats,mean(circle_radius),std(circle_radius),skewness(circle_radius),kurtosis(circle_radius)];
        variable_names={variable_names{1:end},'Mean gap radius','Stan. dev. gap radius','Skewness gap radius','Kurtosis gap radius'};
        percentiles=[0,1,5,10,25,50,75,90,95,99,100];
        for I = 1:length(percentiles)
            p = percentiles(I);
            output_stats = [output_stats,prctile(circle_radius,p)];
            variable_names={variable_names{1:end},['Gap size percentile ' num2str(p) ]};
                 
        end
        h=figure;
        histogram(circle_radius)
        axis;
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Gap size radius (pixels)','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory 'histogram_gap_radius.tif']
        saveas(h,save_filename)
        close all


        output_stats = [output_stats,mean(area_weighted_sample),std(area_weighted_sample),skewness(area_weighted_sample),kurtosis(area_weighted_sample)];
        variable_names={variable_names{1:end},'Mean area weighted gap radius','Stan. dev. area weighted gap radius','Skewness area weighted gap radius','Kurtosis area weighted gap radius'};
        for I = 1:length(percentiles)
            p = percentiles(I);
            output_stats = [output_stats,prctile(area_weighted_sample,p)];
            variable_names={variable_names{1:end},['Area weighted gap size percentile ' num2str(p) ]};
        end
        h=figure;
        histogram(area_weighted_sample)
        axis;
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Gap size radius (pixels)','fontsize',24,'fontweight','b')
        ylabel('Area weighted frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory 'histogram_area_weighted_gap_radius.tif']
        saveas(h,save_filename)
        close all


        output_stats = [output_stats,mean(neighbours(:),'omitnan'),std(neighbours(:),'omitnan'),skewness(neighbours(:)),kurtosis(neighbours(:))];
        variable_names={variable_names{1:end},'Mean gap neighbour size','Stan. dev. gap neighbour size','Skewness gap neighbour size','Kurtosis gap neighbour size'};
        h=figure;
        histogram(neighbours(find(neighbours>=0)))
        axis;
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Neighbour gap size radius (pixels)','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory 'histogram_neighbour_gap_size.tif']
        saveas(h,save_filename)
        close all
        
        for r=1:length(radii_vector)
            output_stats = [output_stats,mean(variable_radius(r).centroid_distance),std(variable_radius(r).centroid_distance),skewness(variable_radius(r).centroid_distance),kurtosis(variable_radius(r).centroid_distance)];
            variable_names={variable_names{1:end},['Mean centroid distance radius threshold ' num2str(radii_vector(r))],['Stan. dev. centroid distance radius threshold ' num2str(radii_vector(r))],['Skewness centroid distance radius threshold ' num2str(radii_vector(r))],['Kurtosis centroid distance radius threshold ' num2str(radii_vector(r))]};
            
            h=figure;
            histogram(variable_radius(r).centroid_distance)
            axis;
            set(gca,'LineWidth',4.5)
            set(gca,'FontSize',20);
            xlabel('Centroid distance (pixels)','fontsize',24,'fontweight','b')
            ylabel('Frequency','fontsize',24,'fontweight','b')
            axis square;
            save_filename=[save_directory 'histogram_gap_centroid_dis_radius' num2str(radii_vector(r)) '.tif']
            saveas(h,save_filename)
            close all
        end
        for r=1:length(radii_vector)
            output_stats = [output_stats,mean(variable_radius(r).boundary_distance),std(variable_radius(r).boundary_distance),skewness(variable_radius(r).boundary_distance),kurtosis(variable_radius(r).boundary_distance)];
            variable_names={variable_names{1:end},['Mean boundary distance radius threshold ' num2str(radii_vector(r))],['Stan. dev. boundary distance radius threshold ' num2str(radii_vector(r))],['Skewness boundary distance radius threshold ' num2str(radii_vector(r))],['Kurtosis boundary distance radius threshold ' num2str(radii_vector(r))]};
            h=figure;
            histogram(variable_radius(r).boundary_distance)
            axis;
            set(gca,'LineWidth',4.5)
            set(gca,'FontSize',20);
            xlabel('Boundary distance (pixels)','fontsize',24,'fontweight','b')
            ylabel('Frequency','fontsize',24,'fontweight','b')
            axis square;
            save_filename=[save_directory 'histogram_gap_boundary_dis_radius' num2str(radii_vector(r)) '.tif']
            saveas(h,save_filename)
            close all
        end
        for I=1:width(inverted_bw_stats)
            shape_stat = inverted_bw_stats.Variables;
            shape_stat = shape_stat(:,I);
            shape_stat(isinf(shape_stat))=[];
            shape_stat(isnan(shape_stat))=[];
            stat_string = inverted_bw_stats.Properties.VariableNames{I}
            output_stats = [output_stats,mean(shape_stat),std(shape_stat),skewness(shape_stat),kurtosis(shape_stat)];
            variable_names={variable_names{1:end},['Mean gap shape ' stat_string],['Skewness ' stat_string],['Stan. dev. ' stat_string],['Kurtosis ' stat_string]};

            h=figure;
            histogram(shape_stat)
            axis;
            set(gca,'LineWidth',4.5)
            set(gca,'FontSize',20);
            xlabel(['Gap shape ' stat_string],'fontsize',24,'fontweight','b')
            ylabel('Frequency','fontsize',24,'fontweight','b')
            axis square;
            save_filename=[save_directory 'histogram_gap_shape_'  stat_string '.tif']
            saveas(h,save_filename)
            close all
        end

        shape_stat = inverted_bw_stats.Area;
        stat_string = 'area'
        output_stats = [output_stats,mean(shape_stat),std(shape_stat),skewness(shape_stat),kurtosis(shape_stat)];
        variable_names={variable_names{1:end},['Mean gap shape ' stat_string],['Skewness gap shape' stat_string],['Stan. dev. gap shape' stat_string],['Kurtosis gap shape' stat_string]};


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Save data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        csv_input=[variable_names;num2cell(output_stats)];
        csv_name=[save_directory,'features_out.csv'];   
        writecell(csv_input',csv_name);
        matlab_name=[save_directory,'features_out.mat'];
        save(matlab_name);
    end

    
end
end




