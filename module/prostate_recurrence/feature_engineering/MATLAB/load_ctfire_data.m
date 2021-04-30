function [fibre_matrix discrete_fibres] = load_ctfire_data(ctfire_file,image_file,minimum_fibre_length)
%LOAD_CTFIRE_DATA Load and transform CTFire output fibre data
%
%   [fibre_matrix discrete_fibres] = 
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
%   fibre_matrix: a matrix of the same dimensions of the tile input
%   image with overlaid fibre information.
%   discrete_fibres: a structure array with x and y coordinates of each
%   fibre.

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
[row_dim col_dim] = size(image);
fibre_matrix=zeros(row_dim,col_dim);
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
    fibre_line=[x_fibre(1:end-1), y_fibre(1:end-1), x_fibre(2:end), y_fibre(2:end)];

    mask=zeros(row_dim, col_dim);
    mask=insertShape(mask,'line',fibre_line,'LineWidth',1);
    mask=mask(:,:,1);
    line_index=find(mask);
    [y_fibre,x_fibre] = ind2sub([row_dim col_dim],line_index);

    discrete_fibres(single_fibre).x=x_fibre;
    discrete_fibres(single_fibre).y=y_fibre;
    x_fibre_all=[x_fibre_all;x_fibre];
    y_fibre_all=[y_fibre_all;y_fibre];
    index_fibre_all=[index_fibre_all;zeros(length(x_fibre),1)+single_fibre];
end

linear_index = sub2ind([row_dim,col_dim], y_fibre_all,x_fibre_all);
fibre_matrix(linear_index)=index_fibre_all;

end