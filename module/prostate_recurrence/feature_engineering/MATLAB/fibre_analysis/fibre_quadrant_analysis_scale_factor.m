function fibre_quadrant_analysis_scale_factor(variable_names,output_stats,quadrant_sf,tile_row,tile_column,total_rows,total_columns,save_directory,ecm_mask,tissue_mask,directory_ctfire_input,ctfire_file,image_folder,image_file,img,angle_continuum,csv_list)
%FIBRE_QUADRANT_ANALYSIS generates fibre feature values for sub-quadrants 
% of the original input tile. 
%
%fibre_quadrant_analysis(variable_names,output_stats,tile_row,tile_column,
% total_rows,total_columns,save_directory,ecm_mask,tissue_mask,
% directory_ctfire_input,ctfire_file,image_folder,image_file,image,
% angle_continuum) generates a csv of fibre
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
%   directory_ctfire_input: Directory where CT-Fire output file is
%   ctfire_file: Filename of Ct_Fire output file
%   image_folder: Location of input image
%   image_file: Input image filename
%   image: Input grayscale deconvolved image for full image.
%   angle_continuum: Continuum angle array
%
%   Output:
%   csv files named with a row-column convention.
%
%
%   Class support for input variable_names:
%      cell
%   Class support for input output_stats, angle_continuum:
%      float: single, double
%   Class support for input ecm_mask, tissue_mask:
%      float: single, double
%   Class support for input tile_row, tile_column, total_rows, 
%   total_columns, discrete_gap_labels,image
%      integer
%   Class support for input save_directory, directory_ctfire_input, 
%   ctfire_file, image_folder, image_file:
%      string
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.

csv_file=['fibre_features_out_quadrant_scalefactor_1_',num2str(quadrant_sf,'%.1d'),'_',num2str(tile_row,'%.2d'),'_',num2str(tile_column,'%.2d'),'.csv'];
csv_name=[save_directory,csv_file];

% Initialize a flag to check if the string is found
string_found = false;
% Loop through the struct array
for i = 1:length(csv_list)
    % Check if the search string is equal to the 'name' field of the current struct
    if strcmp(csv_file, csv_list(i).name)
        % Set the flag to true and break out of the loop
        string_found = true
        i
        csv_list(i).name
        csv_name 
        break;
    end
end


if string_found == true
    'Skipping quadrant. Already processed.'
else


    size(img);
    img;
    row_lb = tile_row * round(total_rows/quadrant_sf) + 1;
    row_ub = min((tile_row+1) * round(total_rows/quadrant_sf),total_rows);
    column_lb = tile_column * round(total_columns/quadrant_sf) + 1;
    column_ub = min((tile_column+1) * round(total_columns/quadrant_sf),total_columns);
    tissue_mask = logical(tissue_mask(row_lb:row_ub,column_lb:column_ub));
    ecm_mask = logical(ecm_mask(row_lb:row_ub,column_lb:column_ub));
    
    img = img(row_lb:row_ub,column_lb:column_ub);
    angle_continuum = angle_continuum(row_lb:row_ub,column_lb:column_ub);
    tissue_proportion = sum(tissue_mask(:))/((row_ub-row_lb+1)*(column_ub-column_lb+1));
    
    if tissue_proportion > 0.2
        directory_ctfire_input, ctfire_file,image_folder, image_file,,5,row_lb,row_ub,column_lb,column_ub
        [ctfire_fibres, fibre_matrix, discrete_fibres] = load_ctfire_data([directory_ctfire_input ctfire_file],[image_folder image_file],5,row_lb,row_ub,column_lb,column_ub);
        fibre_matrix_zero=fibre_matrix;
        fibre_matrix_zero(isnan(fibre_matrix_zero))=0;
        %Run Feature extraction
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %branchpoint, endpoint and shape analysis
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [fibre_skeleton,fibre_branchpoints,fibre_endpoints,fibre_disconnected_branches] = fibre_skeletonisation(fibre_matrix);
        
        %Analyse fibre shapes
        [fibre_shapes] = fibre_shape(ctfire_fibres);
    
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
        if quadrant_sf==2
            radial_distances = round([0.5,1,2,5,10,15,20,30,40]/0.22);
        elseif quadrant_sf==4
            radial_distances = round([0.5,1,2,5,10,15,20]/0.22);
        elseif quadrant_sf == 8
            radial_distances = round([0.5,1,2,5,10]/0.22);
        end
        sample_size=1000;
        random_perm_index = randperm(length(ecm_index),min(sample_size,length(ecm_index)));
        ecm_uniform_random_sample = ecm_index(random_perm_index);
        
        mass=img(ecm_index);
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
              img,...
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
              img,...
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
        output_stats = [output_stats,p];
        variable_names={variable_names{1:end},'ttest_curvature_branchpts_vs_branches'};
        [h,p]=ttest2(curvature_continuum(fibre_branchpoints>0),curvature_continuum(fibre_disconnected_branches>0))
        output_stats = [output_stats,p];
        variable_names={variable_names{1:end},'ttest_curvature_branches_vs_endpoints'};
        [h,p]=ttest2(curvature_continuum(fibre_endpoints>0),curvature_continuum(fibre_disconnected_branches>0))
        output_stats = [output_stats,p];
        g1=ones(1,size(curvature_continuum(fibre_branchpoints>0),1));
        g2=2*ones(1,size(curvature_continuum(fibre_endpoints>0),1));
        g3=3*ones(1,size(curvature_continuum(fibre_disconnected_branches>0),1));
        x=[curvature_continuum(fibre_branchpoints>0);curvature_continuum(fibre_endpoints>0);curvature_continuum(fibre_disconnected_branches>0)]' ;
        g=[g1,g2,g3]; 
        p=anova1(x,g,'off');
        variable_names={variable_names{1:end},['anova_curvature_branchpoints_vs_endpoints_vs_branches']};
        output_stats = [output_stats,p];        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Spatial derivatives
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        [curvature_X,curvature_Y] = gradient(curvature_continuum); 
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
    
        input_gap_matrix = ecm_mask;
    
        %HDM normalised to mask area
        variable_names={variable_names{1:end},'total_tissue_proportion'};
        output_stats = [output_stats,sum(tissue_mask(:))/((row_ub-row_lb+1)*(column_ub-column_lb+1))];
        HDM_fibre=length(find(fibre_matrix>0))/sum(tissue_mask(:));
        variable_names={variable_names{1:end},'HDM_fibre'};
        output_stats = [output_stats,HDM_fibre];
        HDM_ecm=sum(input_gap_matrix(:))/sum(tissue_mask(:));
        variable_names={variable_names{1:end},'HDM_ecm'};
        output_stats = [output_stats,HDM_ecm];
        HDM_fibre_ecm_ratio=length(find(fibre_matrix>0))/(sum(input_gap_matrix(:)));
        variable_names={variable_names{1:end},'HDM_fibre_ecm_ratio'};
        output_stats = [output_stats,HDM_fibre_ecm_ratio];
        %Fractal dimension
        
        L = bwlabel(input_gap_matrix,8);
        keep_labels = unique(logical(fibre_matrix_zero).*L);
        [C,ia] = setdiff(L,keep_labels,'sorted');
        L_remove = ismember(L,C);
        index_remove=find(L_remove);
        L1=L;
        L1(index_remove)=0;
        input_gap_matrix = logical(L1);
        if quadrant_sf==2
            fractal_range=[1,4,16,64,256];
        elseif quadrant_sf==4
            fractal_range=[1,4,16,64,256];
        elseif quadrant_sf == 8
            fractal_range=[1,4,16,64];
        end
        [n,r] = boxcount(input_gap_matrix,'slope');
        x=log(r);
        y=log(n);
        dy=-gradient(y)./gradient(x);
        mdl = fitlm(r(1:end),dy(1:end),'constant','RobustOpts','on')
        coeffs = mdl.Coefficients.Estimate;
        variable_names={variable_names{1:end},['fractal_dimension_whole']};
        output_stats = [output_stats,coeffs(1)];
        for I=1:length(fractal_range)-1
            fractal_range(I)
            lb=find(r==fractal_range(I));
            ub=find(r==fractal_range(I+1));
            mdl = fitlm(r(lb:ub),dy(lb:ub),'constant','RobustOpts','on')
            coeffs = mdl.Coefficients.Estimate;
            variable_names={variable_names{1:end},['fractal_dimension_range_' num2str(r(lb)) '_to_' num2str(r(ub))]};
            output_stats = [output_stats,coeffs(1)];
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Save data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        csv_input=[variable_names;num2cell(output_stats)];
        csv_name=[save_directory,'fibre_features_out_quadrant_scalefactor_1_',num2str(quadrant_sf,'%.1d'),'_',num2str(tile_row,'%.2d'),'_',num2str(tile_column,'%.2d'),'.csv'];   
        writecell(csv_input',csv_name);
    end
end
end