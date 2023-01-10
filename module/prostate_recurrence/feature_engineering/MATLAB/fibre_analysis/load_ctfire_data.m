function [ctfire_fibres,fibre_matrix,discrete_fibres] = ...
    load_ctfire_data(ctfire_file,...
    image_file,...
    minimum_fibre_length,...
    row_minimum,...
    row_maximum,...
    column_minimum,...
    column_maximum)
%LOAD_CTFIRE_DATA Load and transform CTFire output fibre data
%
%   [ctfire_fibres fibre_matrix discrete_fibres] = 
%   load_ctfire_data(ctfire_file,image_file,minimum_fibre_length,
%   row_minimum,row_maximum,column_minimum,column_maximum) loads
%   fibre level information from the CT-Fire output, creates connected
%   object fibres from the discontinuous CTFire fibre points and outputs
%   the data in a matrix the same size as the tile and as a structure 
%   array. Only fibres in a subregion given by the row and column mimimums 
%   and maximums are considered and the output data translated to fit on a 
%   tile of row size row_maximum - row_minimum + 1 and equivalent for 
%   columns is considered. This allows processing of both the original tile
%   and sub-quadrants.
%
%   Input:
%   ctfire_file: CTFire output file. 
%   image_file: corresponding input tile image.
%   minimum_fibre_length: a single value that gives the minimum pixel 
%   length fibre included in the fibre output. Lengths of each fibre 
%   are given in CTFire output
%   row_minimum: Minimum row to consider fibres in (used to define whther
%   processing a full tile or suquadrant).
%   row_maximum: Maximum row to consider fibres in.
%   column_minimum: Minimum column to consider fibres in.
%   column_maximum: Maximum column to consider fibres in.
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
%   Class support for row_minimum, row_maximum, column_minimum, 
%   column_maximum:
%      float: single, double, int: uint8, uint16, uint64
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.

image=imread(image_file);
fibre_matrix=zeros(row_maximum-row_minimum+1,column_maximum-column_minimum+1)+NaN;
[row_dim, col_dim] = size(fibre_matrix);
load(ctfire_file);
fibre_threshold_index = find(data.M.L >= minimum_fibre_length);
number_fibres = length(fibre_threshold_index);
x_fibre_all=[];
y_fibre_all=[];
index_fibre_all=[];
counter=0;
for single_fibre = 1:number_fibres
    single_fibre_index = data.Fa(1,fibre_threshold_index(single_fibre)).v;
    fibre_coords = data.Xa(single_fibre_index,:);
    x_fibre=fibre_coords(:,1);
    y_fibre=fibre_coords(:,2);

    index_x_lb=find(x_fibre>=column_minimum-0.5);
    index_x_ub=find(x_fibre<column_maximum+0.5);
    index_y_lb=find(y_fibre>=row_minimum-0.5);
    index_y_ub=find(y_fibre<row_maximum+0.5);
    index = intersect(intersect(index_x_lb,index_x_ub),intersect(index_y_lb,index_y_ub));
    x_fibre = x_fibre(index)-column_minimum;
    y_fibre = y_fibre(index)-row_minimum;
    if length(x_fibre)>1
        counter=counter+1;
        ctfire_fibres(counter).x=x_fibre;
        ctfire_fibres(counter).y=y_fibre;

    
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

    discrete_fibres(counter).x=x_fibre;
    discrete_fibres(counter).y=y_fibre;
    x_fibre_all=[x_fibre_all;x_fibre];
    y_fibre_all=[y_fibre_all;y_fibre];
    index_fibre_all = ...
        [index_fibre_all;zeros(length(x_fibre),1)+counter];
    
    %For the continuous fibre form, discrete_fibre, find the closest point
    %from ctfire_fibres so that we impose angle information from 
    %ctfire_fibres to the closest point in discrete_fibres
    discrete_x_mesh = ...
        repmat(...
        discrete_fibres(counter).x,...
        1,...
        length(ctfire_fibres(counter).x)...
        );
    discrete_y_mesh = ...
        repmat(...
        discrete_fibres(counter).y,...
        1,...
        length(ctfire_fibres(counter).y)...
        );
    ctfire_x_mesh = ...
        repmat(...
        ctfire_fibres(counter).x,...
        1,...
        length(discrete_fibres(counter).x)...
        )';
    ctfire_y_mesh = ...
        repmat(...
        ctfire_fibres(counter).y,...
        1,...
        length(discrete_fibres(counter).y)...
        )';
    distance_transform = ...
        (...
        (discrete_x_mesh-ctfire_x_mesh).^2+...
        (discrete_y_mesh-ctfire_y_mesh).^2 ...
        ).^0.5;
    [~,ctfire_point] = min(distance_transform,[],2);
    discrete_fibres(counter).ctfire_point = ctfire_point;
    end
    
end

linear_index = sub2ind([row_dim,col_dim], y_fibre_all,x_fibre_all);
fibre_matrix(linear_index)=index_fibre_all;

end