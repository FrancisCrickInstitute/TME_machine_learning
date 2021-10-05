function [ctfire_fibres,fibre_matrix,discrete_fibres] = ...
    load_ctfire_data(ctfire_file,image_file,minimum_fibre_length)
%LOAD_CTFIRE_DATA Load and transform CTFire output fibre data
%
%   [ctfire_fibres fibre_matrix discrete_fibres] = 
%   load_ctfire_data(ctfire_file,image_file,minimum_fibre_length) loads
%   fibre level information from the CT-Fire output, creates connected
%   object fibres from the discontinuous CTFire fibre points and outputs
%   the data in a matrix the same size as the tile and as a structure array
%   underlying function V=F(X) at the query points Xq. 
%
%   Input:
%   ctfire_file: CTFire output file. 
%   image_file: corresponding input tile image.
%   minimum_fibre_length: a single value that gives the minimum pixel 
%   length fibre included in the fibre output. Lengths of each fibre 
%   are given in CTFire output
%
%   Output:
%   ctfire_fibres: a structure array with x and y coordinates of each fibre
%   as in the CTFire output%
%   fibre_matrix: a matrix of the same dimensions of the tile input
%   image with overlaid fibre information.
%   discrete_fibres: a structure array with x and y coordinates of each
%   fibre to account for discontinuities. It also includes a distance
%   mapping showing which point in the original discontinuous CTFire output
%   vectors, each continous point in discrete_fibres is closest to. This is
%   used to map angle and curvature information from the CTFire output onto
%   the continuous fibres.
%
%   Individual CTFire output fibres are composed of discontinuous points. 
%   To create corresponding discrete connected fibre objects, a mask of
%   each fibre is created using MATLAB's insertShape function to create an
%   image mask of each continuous line. 
%
%   Class support for inputs ctfire_file,image_file:
%      string: string
%   Class support for input minimum_fibre_length:
%      float: single, double, int: uint8, uint16, uint64
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.

image=imread(image_file);
[row_dim, col_dim] = size(image);
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
    ctfire_fibres(single_fibre).x=x_fibre;
    ctfire_fibres(single_fibre).y=y_fibre;
    
    fibre_line = ...
        [...
        x_fibre(1:end-1),...
        y_fibre(1:end-1),...
        x_fibre(2:end),...
        y_fibre(2:end)...
        ];
    mask=zeros(row_dim, col_dim);
    mask=insertShape(mask,'line',fibre_line,'LineWidth',1);
    mask=mask(:,:,1);
    line_index=find(mask);
    [y_fibre,x_fibre] = ind2sub([row_dim col_dim],line_index);

    discrete_fibres(single_fibre).x=x_fibre;
    discrete_fibres(single_fibre).y=y_fibre;
    x_fibre_all=[x_fibre_all;x_fibre];
    y_fibre_all=[y_fibre_all;y_fibre];
    index_fibre_all = ...
        [index_fibre_all;zeros(length(x_fibre),1)+single_fibre];
    
    %For the continuous fibre form, discrete_fibre, find the closest point
    %from ctfire_fibres so that we impose angle information from 
    %ctfire_fibres to the closest point in discrete_fibres
    discrete_x_mesh = ...
        repmat(...
        discrete_fibres(single_fibre).x,...
        1,...
        length(ctfire_fibres(single_fibre).x)...
        );
    discrete_y_mesh = ...
        repmat(...
        discrete_fibres(single_fibre).y,...
        1,...
        length(ctfire_fibres(single_fibre).y)...
        );
    ctfire_x_mesh = ...
        repmat(...
        ctfire_fibres(single_fibre).x,...
        1,...
        length(discrete_fibres(single_fibre).x)...
        )';
    ctfire_y_mesh = ...
        repmat(...
        ctfire_fibres(single_fibre).y,...
        1,...
        length(discrete_fibres(single_fibre).y)...
        )';
    distance_transform = ...
        (...
        (discrete_x_mesh-ctfire_x_mesh).^2+...
        (discrete_y_mesh-ctfire_y_mesh).^2 ...
        ).^0.5;
    [~,ctfire_point] = min(distance_transform,[],2);
    discrete_fibres(single_fibre).ctfire_point = ctfire_point;
    
end

linear_index = sub2ind([row_dim,col_dim], y_fibre_all,x_fibre_all);
fibre_matrix(linear_index)=index_fibre_all;

end