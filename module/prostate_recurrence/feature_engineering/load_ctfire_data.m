function [fibre_matrix discrete_fibres] = load_ctfire_data(ctfire_file,image_file,minimum_fibre_length)

    image=imread(image_file);
    [row_dim col_dim] = size(image);
    fibre_matrix=zeros(row_dim,col_dim)+NaN;
    
    load(ctfire_file);
    fibre_threshold_index = find(data.M.L >= minimum_fibre_length);
    number_fibres = length(fibre_threshold_index);
    
    x_fibre_all=[];
    y_fibre_all=[];
    index_fibre_all=[];
    for single_fibre = 1:number_fibres
        single_fibre_index = data.Fa(1,fibre_threshold_index(single_fibre)).v;
        fibre_coords = data.Xa(single_fibre_index,:);
        x_fibre=fibre_coords(:,1);
        y_fibre=fibre_coords(:,2);
        discrete_fibres(single_fibre).x=x_fibre;
        discrete_fibres(single_fibre).y=y_fibre;
        x_fibre_all=[x_fibre_all;x_fibre];
        y_fibre_all=[y_fibre_all;y_fibre];
        index_fibre_all=[index_fibre_all;zeros(length(x_fibre),1)+single_fibre];
    end
    
    linear_index = sub2ind([row_dim,col_dim], y_fibre_all,x_fibre_all);
    fibre_matrix(linear_index)=index_fibre_all;
end