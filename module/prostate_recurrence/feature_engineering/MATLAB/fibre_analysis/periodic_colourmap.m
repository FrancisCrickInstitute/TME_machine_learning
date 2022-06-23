function periodic_colourmap(...
    directory,...
    file_name,...
    file_type,...
    periodic_matrix,...
    include_colourbar,...
    tissue_mask...
    )
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
%   tissue_mask: Values beyond the tissue mask are set to 0 (i.e. black on
%   the output colourmap.
%
%
%   Class support for inputs directory, file_name, file_type:
%      string
%   Class support for inputs periodic_matrix and include_colourbar:
%      float: single, double, int: uint8, uint16, uint64
%   Class support for tissue_mask:
%       logical
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.


hmap(1:256,1) = linspace(0,1,256);
hmap(:,[2 3]) = 0.8; %brightness
huemap = hsv2rgb(hmap);
huemap=[huemap;[1,1,1]];
periodic_index = round(255*(periodic_matrix-0)./pi)+1;
rgb_image = ind2rgb(periodic_index, huemap);
tissue_mask = repmat(tissue_mask,[1,1,3]);
rgb_image(~tissue_mask) = 0;

if file_type(1) ~= '.'
    file_type=['.' file_type];
end
if directory(end)~='/'
    directory=[directory '/'];
end
image_name = [directory file_name file_type];
imwrite(uint8(255*rgb_image),huemap,image_name);

if include_colourbar == 1
    ax = axes;
    x=0:0.25:1;
    colormap(huemap)
    c = colorbar(...
        'Ticks',x,...
        'TickLabels',{'0','\pi/4','\pi/2','3\pi/4','\pi'},...
        'LineWidth',3,...
        'FontSize',20,...
        'Location','west'...
        );
    c.Label.String = 'Angle (radians)';
    c.Label.FontWeight = 'bold';
    ax.Visible = 'off';
    colourbar_name = [directory 'hsv_periodic_colourbar' file_type];
    saveas(ax,colourbar_name);
    close all
end
