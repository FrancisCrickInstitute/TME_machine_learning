function linear_colourmap(directory,file_name,file_type,input_matrix,colourmap,include_colourbar,colourbar_name,max_input,min_input)
%PERIODIC_COLOURMAP Saves periodic input matrix using hsv wraparound
%colourmap
%
%   periodic_colourmap(directory,file_name,file_type,periodic_matrix,
%   include_colourbar) hsv colourmap of the input angle matrix (discrete or
%   continuum, end to end or local). A colourbar with corresponding radian 
%   colours is also optionally generated. 
%
%   Input:
%   directory: Location where colourmaps are saved.
%   file_name: Save name of colourmap.
%   file_type: Save image type (tif, jpeg etc.)
%   periodic_matrix: The matrix of angles used to generate image output.
%   include_colourbar: 1 for True, 0 for false.
%
%
%   Class support for inputs directory, file_name, file_type:
%      string
%   Class support for inputs periodic_matrix and include_colourbar:
%      float: single, double, int: uint8, uint16, uint64
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.


cm=[colourmap '(256)'];
cm=eval(cm);
cm=[cm;[1,1,1]];
if length(max_input)==0
    max_input = max(input_matrix(:));
end
if length(min_input)==0
    min_input = min(input_matrix(:));
end
input_matrix(input_matrix>max_input)=max_input;
input_matrix(input_matrix<min_input)=min_input;
linear_index = round(255*(input_matrix-min_input)./(max_input-min_input))+1;
rgb_image = ind2rgb(linear_index, cm);

if file_type(1) ~= '.'
    file_type=['.' file_type];
end
if directory(end)~='\'
    directory=[directory '\'];
end
image_name = [directory file_name file_type];
imwrite(uint8(255*rgb_image),cm,image_name);

if include_colourbar == 1
    ax = axes;
    x=0:0.25:1;
    colormap(cm)
    c = colorbar('Ticks',x,'TickLabels',min_input:(max_input-min_input)/4:max_input,'LineWidth',3,'FontSize',20,'Location','west');
    %c.Label.String = 'Angle (radians)';
    c.Label.FontWeight = 'bold';
    ax.Visible = 'off';
    colourbar_name = [directory colourbar_name file_type];
    saveas(ax,colourbar_name);
end
close all
