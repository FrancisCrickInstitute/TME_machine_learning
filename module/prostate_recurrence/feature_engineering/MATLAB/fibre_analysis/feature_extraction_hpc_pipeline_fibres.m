function feature_extraction_hpc_pipeline_fibres(image_start,image_end,image_folder,tissue_folder,function_folder,ctfire_function_folder,ecm_threshold)

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
    ecm_mask = image;
    ecm_mask(ecm_mask<=ecm_threshold)=0;
    ecm_mask(ecm_mask>ecm_threshold)=1;
    ecm_mask = logical(logical(ecm_mask).*tissue_mask);
    ecm_mask = bwareaopen(ecm_mask,20,8);

    if sum(ecm_mask(:))/length(tissue_mask(:)) >= 0.005%Threshold to accept tile
        %Run CT-Fire
        
        image_folder
        output_folder = [strrep(image_folder,'pre_processed_data','feature_engineering'),'tile_level_features/',name '/'];
        str1 = strfind(output_folder,'/');
        str2 = strfind(output_folder,'feature_engineering/');
        str_index=str1(str1>str2);
        for str_i=1:length(str_index)
           directory_create=output_folder(1:str_index(str_i))
           mkdir(directory_create)
        end



        output_image_dir = [output_folder '*' image_type];
        fileList = dir(output_image_dir);
        for f=1:length(fileList)
            k = strfind(fileList(f).name,'gap');
            if isempty(k)
                file_delete = [output_folder fileList(f).name];
                delete(file_delete);
            end
        end


        %delete(output_image_dir)
        output_csv_dir = [output_folder 'fibre*.csv' ];
        delete(output_csv_dir)
        output_mat_dir = [output_folder 'fibre*.mat' ];
        delete(output_mat_dir)

        imageRange = num2str(image_I); % 'all': default, all images in the image folder ; or specified image indexes, such as '1:2', or '1:2 3'
        cd(ctfire_function_folder);
        CurveAlign_CommandLine(image_folder,image_type,analysis_mode,imageRange);
        cd([current_dir '/']);
        
        %Load CT-FIRE
        save_directory = [output_folder '/' name '_automated_results/'];
        save_directory = output_folder;
        mkdir(save_directory);
        input_ctfire_dir = [directory_ctfire_input '*' name '*' '.mat'];
        input_ctfire_list = dir(input_ctfire_dir);
        ctfire_file=input_ctfire_list(1).name;



        %[ctfire_fibres fibre_matrix discrete_fibres] = load_ctfire_data([directory_ctfire_input ctfire_file],[image_folder image_file],5);
        [row_dim, col_dim] = size(image);
        [ctfire_fibres,fibre_matrix,discrete_fibres] = load_ctfire_data([directory_ctfire_input ctfire_file],[image_folder image_file],5,1,row_dim,1,col_dim);
        fibre_matrix_zero=fibre_matrix;
        fibre_matrix_zero(isnan(fibre_matrix_zero))=0;
        

        %Run Feature extraction

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %branchpoint, endpoint and shape analysis
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [fibre_skeleton,fibre_branchpoints,fibre_endpoints,fibre_disconnected_branches] = fibre_skeletonisation(fibre_matrix);
        

        
        %Analyse fibre shapes
        [fibre_shapes] = fibre_shape(ctfire_fibres);%Need ctfire_fibres expressed for each quantiles

        branch_labels = bwlabel(fibre_branchpoints,8);
        endpoints_labels = bwlabel(fibre_endpoints,8);
        branch_pts_per_pixel = max(branch_labels(:))/sum([fibre_shapes.length]);
        end_pts_per_pixel = max(endpoints_labels(:))/sum([fibre_shapes.length]);
        variable_names={variable_names{1:end},'branch_points','end_points'};
        output_stats = [output_stats,branch_pts_per_pixel,end_pts_per_pixel];


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

        variable_names={variable_names{1:end},'mean_fibre_length','std_fibre_length','skew_fibre_length','kurtosis_fibre_length','mean_fibre_displacement','std_fibre_displacement','skew_fibre_displacement','kurtosis_fibre_displacement',...
            'mean_fibre_persistence','std_fibre_persistence','skew_fibre_persistence','kurtosis_fibre_persistence','mean_fibre_area','std_fibre_area','skew_fibre_area','kurtosis_fibre_area',...
            'mean_fibre_circularity','std_fibre_circularity','skew_fibre_circularity','kurtosis_fibre_circularity',...
            'mean_fibre_length/perim.','std_fibre_length/perim.','skew_fibre_length/perim.','kurtosis_fibre_length/perim.'};
        output_stats = [output_stats,Shape_stats(1,1),Shape_stats(2,1),Shape_stats(3,1),Shape_stats(4,1),Shape_stats(1,2),Shape_stats(2,2),Shape_stats(3,2),Shape_stats(4,2),Shape_stats(1,3),Shape_stats(2,3),Shape_stats(3,3),Shape_stats(4,3),Shape_stats(1,4),Shape_stats(2,4),Shape_stats(3,4),Shape_stats(4,4),Shape_stats(1,5),Shape_stats(2,5),Shape_stats(3,5),Shape_stats(4,5),Shape_stats(1,6),Shape_stats(2,6),Shape_stats(3,6),Shape_stats(4,6)];



        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %discrete and continuum curvature analysis
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [discrete_fibre_curvature,curvature_matrix,curvature_continuum] = fibre_curvature(ctfire_fibres, discrete_fibres, fibre_matrix);
        curvature=[];
        for I=1:length(discrete_fibre_curvature)
            if max(isnan(discrete_fibre_curvature(I).curvature))==0
                curvature=[curvature;discrete_fibre_curvature(I).curvature];
            end
        end


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
        





        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %discrete and continuum angle analysis
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [discrete_fibre_angles,end_to_end_angle_matrix,end_to_end_x_derivative_matrix,end_to_end_y_derivative_matrix,angle_matrix,local_x_derivative_matrix,local_y_derivative_matrix] = fibre_angle(ctfire_fibres, discrete_fibres, fibre_matrix);
        [end_to_end_angle_continuum,end_to_end_x_derivative_continuum,end_to_end_y_derivative_continuum,angle_continuum,local_x_derivative_continuum,local_y_derivative_continuum] = fibre_angle_continuum(end_to_end_x_derivative_matrix,end_to_end_y_derivative_matrix,local_x_derivative_matrix,local_y_derivative_matrix);

        %Circular_stats
        angles=[];
        for I=1:length(discrete_fibre_angles)
            angles=[angles;discrete_fibre_angles(I).local_fibre_angle];
        end
        local_circ_stats=circ_stats(2*angles);


        variable_names={variable_names{1:end},'angle_var','angle_std','angle_std0','angle_skew','angle_skew0','angle_kurtosis','angle_kurtosis0'};
        output_stats = [output_stats,local_circ_stats.var,local_circ_stats.std,local_circ_stats.std0,local_circ_stats.skewness,local_circ_stats.skewness0,local_circ_stats.kurtosis,local_circ_stats.kurtosis0];
 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %fibre alignment over length scales
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        ecm_index=find(ecm_mask);
        radial_distances = round([0.5,1,2,5,10,15,20,30,40,50,60,80]/0.22);
        sample_size=1000;
        random_perm_index = randperm(length(ecm_index),sample_size);
        ecm_uniform_random_sample = ecm_index(random_perm_index);
        
        mass=image(ecm_index);
        mass=mass-min(mass)+1;
        mass=double(mass)/sum(mass(:));
        cum_mass=cumsum(mass);
        U=rand(1000,1);
        for u=1:length(U)
        weighted_random_perm_index(u) = find(cum_mass>U(u),1);
        end
        ecm_weighted_random_sample = ecm_index(weighted_random_perm_index);
        
        [row_dim,col_dim]=size(ecm_mask);
        for r=1:length(radial_distances)
            radial_distance=radial_distances(r);
            [angle_difference,weighting_ecm] = random_alignment_quantification(...
              radial_distance,...
              col_dim,...
              row_dim,...
              ecm_index,...
              ecm_uniform_random_sample,...
              image,...
              angle_continuum);
            output_stats = [output_stats,mean(angle_difference)];
            variable_names={variable_names{1:end},['mean_uniform_sampled_alignment_pixel_radius_' num2str(radial_distance) ]};
            output_stats = [output_stats,std(angle_difference)];
            variable_names={variable_names{1:end},['std_uniform_sampled_alignment_pixel_radius_' num2str(radial_distance) ]};
            output_stats = [output_stats,skewness(angle_difference)];
            variable_names={variable_names{1:end},['skewness_uniform_sampled_alignment_pixel_radius_' num2str(radial_distance) ]};
            output_stats = [output_stats,kurtosis(angle_difference)];
            variable_names={variable_names{1:end},['kurtosis_uniform_sampled_alignment_pixel_radius_' num2str(radial_distance) ]};
        end
        
        for r=1:length(radial_distances)
            radial_distance=radial_distances(r);
            [angle_difference,weighting_ecm] = random_alignment_quantification(...
              radial_distance,...
              col_dim,...
              row_dim,...
              ecm_index,...
              ecm_weighted_random_sample,...
              image,...
              angle_continuum);
            output_stats = [output_stats,sum(weighting_ecm.*angle_difference)./sum(weighting_ecm)];
            variable_names={variable_names{1:end},['mean_weighted_sampled_alignment_pixel_radius_' num2str(radial_distance) ]};
            output_stats = [output_stats,(var(angle_difference,weighting_ecm))^0.5];
            variable_names={variable_names{1:end},['std_weighted_sampled_alignment_pixel_radius_' num2str(radial_distance) ]};
        end



        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %branchpoint, endpoint and branches overlay on continuum fields
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        
        %Overlay branchpoints and endpoints over curvature
        branches_curvature = curvature_continuum;
        branches_curvature(fibre_branchpoints>0) = NaN;

        endpoints_curvature = curvature_continuum;
        endpoints_curvature(fibre_endpoints>0) = NaN;

        branches_curvature = curvature_continuum;
        branches_curvature(fibre_disconnected_branches>0) = NaN;


        variable_names={variable_names{1:end},'ttest_curvature_branchpts_vs_endpoints'};
        [h,p]=ttest2(curvature_continuum(fibre_branchpoints>0),curvature_continuum(fibre_endpoints>0))
        output_stats = [output_stats,log10(p)];
        variable_names={variable_names{1:end},'ttest_curvature_branchpts_vs_branches'};
        [h,p]=ttest2(curvature_continuum(fibre_branchpoints>0),curvature_continuum(fibre_disconnected_branches>0))
        output_stats = [output_stats,log10(p)];
        variable_names={variable_names{1:end},'ttest_curvature_branches_vs_endpoints'};
        [h,p]=ttest2(curvature_continuum(fibre_endpoints>0),curvature_continuum(fibre_disconnected_branches>0))
        output_stats = [output_stats,log10(p)];
        g1=ones(1,size(curvature_continuum(fibre_branchpoints>0),1));
        g2=2*ones(1,size(curvature_continuum(fibre_endpoints>0),1));
        g3=3*ones(1,size(curvature_continuum(fibre_disconnected_branches>0),1));
        x=[curvature_continuum(fibre_branchpoints>0);curvature_continuum(fibre_endpoints>0);curvature_continuum(fibre_disconnected_branches>0)]' ;
        g=[g1,g2,g3]; 
        p=anova1(x,g,'off');
        variable_names={variable_names{1:end},['anova_curvature_branches_vs_endpoints_vs_branchpts']};
        output_stats = [output_stats,log10(p)];        
 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Spatial derivatives
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        [curvature_X,curvature_Y] = gradient(curvature_continuum);

        save_filename = 'curvature_X_derivative';
        max_grad=max(max(curvature_X(:)),max(curvature_Y(:)))
        min_grad=min(min(curvature_X(:)),min(curvature_Y(:)))
        plotted_matrix = curvature_X; 
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',1,'Curvature_x_derivative_cm',0.05,-0.05,ecm_mask);
        save_filename = 'curvature_Y_derivative'; 
        plotted_matrix = curvature_Y; 
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',1,'Curvature_x_derivative_cm',0.05,-0.05,ecm_mask);

        curvature_X = curvature_X(ecm_mask);
        curvature_Y = curvature_Y(ecm_mask);
        curvature_derivative_correlation = corr(curvature_X(:),curvature_Y(:));

        variable_names={variable_names{1:end},'curvature_derivative_correlation'};
        output_stats = [output_stats,curvature_derivative_correlation];



        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Randomness of spatial distribution of curvature and alignment
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %Are distributions of curvature etc. uniformly spatially distributed?
        %Looking at uniform distribution  
        Ufm=ecm_mask;
        Ufm=Ufm./sum(Ufm(:));
        [rows,cols]=size(fibre_matrix);
        [x,y] = meshgrid(1:cols,1:rows);
        curvature_dis=curvature_continuum.*ecm_mask;
        curvature_dis=curvature_dis./sum(curvature_dis(:));

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
        jensen_shannon=(0.5*sum(ys1)+0.5*sum(ys2))^0.5;
        bhattacharyya=sum((Z1(:).*Z2(:)).^0.5);
        hellinger=(sum((Z1.^0.5-Z2.^0.5).^2))^0.5/(2^0.5);
        distribution_distance(1,1)=jensen_shannon;
        distribution_distance(2,1)=bhattacharyya;
        distribution_distance(3,1)=hellinger;
        variable_names={variable_names{1:end},'curvature_jensen_shannon', 'curvature_bhattacharyya', 'curvature_hellinger'};
        output_stats = [output_stats,jensen_shannon,bhattacharyya,hellinger];



        %HDM Features

        %HDM normalised to mask area
        variable_names={variable_names{1:end},'total_tissue_proportion'};
        output_stats = [output_stats,sum(tissue_mask(:))/(row_dim*col_dim)];
        HDM_fibre=length(find(fibre_matrix>0))/sum(tissue_mask(:));
        variable_names={variable_names{1:end},'HDM_fibre'};
        output_stats = [output_stats,HDM_fibre];
        HDM_ecm=sum(ecm_mask(:))/sum(tissue_mask(:));
        variable_names={variable_names{1:end},'HDM_ecm'};
        output_stats = [output_stats,HDM_ecm];
        HDM_fibre_ecm_ratio=length(find(fibre_matrix>0))/(sum(ecm_mask(:)));
        variable_names={variable_names{1:end},'HDM_fibre_ecm_ratio'};
        output_stats = [output_stats,HDM_fibre_ecm_ratio];

        %Fractal dimension
        
        L = bwlabel(ecm_mask,8);
        keep_labels = unique(logical(fibre_matrix_zero).*L);
        [C,ia] = setdiff(L,keep_labels,'sorted');
        L_remove = ismember(L,C);
        index_remove=find(L_remove);
        L1=L;
        L1(index_remove)=0;
        input_gap_matrix = logical(L1);

        fractal_range=[1,4,16,64,256,1024];
        [n,r] = boxcount(input_gap_matrix,'slope');
        x=log(r);
        y=log(n);
        dy=-gradient(y)./gradient(x);
        mdl = fitlm(r(1:end),dy(1:end),'constant','RobustOpts','on')
        coeffs = mdl.Coefficients.Estimate;
        variable_names={variable_names{1:end},['fractal_dimension_whole']};
        output_stats = [output_stats,coeffs(1)];

        for I=1:length(fractal_range)-1
            lb=find(r==fractal_range(I));
            ub=find(r==fractal_range(I+1));
            mdl = fitlm(r(lb:ub),dy(lb:ub),'constant','RobustOpts','on')
            coeffs = mdl.Coefficients.Estimate;
            variable_names={variable_names{1:end},['fractal_dimension_range_' num2str(r(lb)) '_to_' num2str(r(ub))]};
            output_stats = [output_stats,coeffs(1)];

        end
        
        WeAreHere=99
        save_directory
        
        %Plot discrete fibres
        fibre_matrix_zero=fibre_matrix;
        fibre_matrix_zero(isnan(fibre_matrix_zero))=0;
        save_filename = 'label_image_discrete';
        plotted_matrix = fibre_matrix;
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'parula',0,'fibre_label',[],[],ecm_mask);


        branchpoint_image = fibre_matrix;
        branchpoint_image(fibre_matrix>0)=1;
        branchpoint_image(fibre_branchpoints>0)=2;
        branchpoint_image(fibre_endpoints>0)=3;
        save_filename = 'branch_image';
        plotted_matrix = branchpoint_image;
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'cool',0,'branch_point_label',[],[],ecm_mask);


        save_filename = 'curvature_discrete';
        plotted_matrix = curvature_matrix; 
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',1,'Curvature',0.5,0,ecm_mask);
        save_filename = 'curvature_continuum';
        plotted_matrix = curvature_continuum;
        linear_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,'jet',1,'Curvature',0.5,0,ecm_mask);

        save_filename = 'angle_discrete';
        plotted_matrix = angle_matrix;
        periodic_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,1,ecm_mask);
        save_filename = 'angle_continuum'; 
        plotted_matrix = angle_continuum;
        periodic_colourmap(save_directory,save_filename,output_filetype,plotted_matrix,1,ecm_mask);

        h=figure;
        histogram([angles])
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('Angle (radians)','fontsize',24,'fontweight','b')
        ylabel('Frequency','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory,'angles_histogram.tif'];
        saveas(h,save_filename)
        close all

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
        semilogx(r, dy, 'k', 'lineWidth',3)
        axis;
        set(gca,'LineWidth',4.5)
        set(gca,'FontSize',20);
        xlabel('${\varepsilon}$, box size (pixels)','interpreter','latex','fontsize',24,'fontweight','b')
        ylabel('${-d\ln(N(\varepsilon))/d\ln(\varepsilon)}$','interpreter','latex','fontsize',24,'fontweight','b')
        axis square;
        save_filename=[save_directory 'fractal_semilog_derivative_plot.tif']
        saveas(h,save_filename)
        close all





        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Save data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        csv_input=[variable_names;num2cell(output_stats)];
        csv_name=[save_directory,'fibre_features_out.csv'];   
        writecell(csv_input',csv_name);
        matlab_name=[save_directory,'fibre_features_out.mat'];


       clearvars -except variable_names output_stats tile_row tile_column row_dim col_dim...
            save_directory ecm_mask tissue_mask directory_ctfire_input ctfire_file...
            image_folder image_file image angle_continuum ctfire_fibres fibre_matrix...
            discrete_fibres fibre_skeleton fibre_branchpoints fibre_endpoints...
            fibre_disconnected_branches fibre_shapes discrete_fibre_curvature...
            curvature_matrix curvature_continuum discrete_fibre_angles angle_matrix...
            angles matlab_name


        save(matlab_name);


        for tile_row = 0:1
            for tile_column = 0:1
                variable_names={};
                output_stats = [];
                fibre_quadrant_analysis(variable_names,output_stats,tile_row,tile_column,row_dim,col_dim,save_directory,ecm_mask,tissue_mask,directory_ctfire_input,ctfire_file,image_folder,image_file,image,angle_continuum);
            end
        end



    end

    
end
end




