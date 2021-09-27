function [fibre_shapes] = fibre_shape(ctfire_fibres)

%FIBRE_SHAPE Calculates features related to shape and shape of region of 
% influence of fibres. Transform fibre network to fibre skeleton to allow
%graph theory based analysis of fibre matrix 
%
%   [fibre_shapes] = fibre_shape(ctfire_fibres) calculates features related
%   to length and straightness of fibre. Additionally carries out Delaunay
%   triangulation to generate alphaShapes from which features related to 
%   shape of region of fibre influence can be generated.
%
%   Input:
%   ctfire_fibres: Structure array of discrete CTFire fibres. 
%
%   Output:
%   fibre_shapes: A structure array of various shape features for each
%   individual fibre. This includes: 
%   length - the total length of a fibre
%   displacement - the distance between the start and end of a fibre
%   persistence - a measure of straightness (displacement/length) 
%   area - total area of alphaShape reflecting area of influence of fibre
%   perimeter - total perimeter of alphaShape reflecting perimeter of 
%   influence of fibre
%   circularity - circularity of alphaShape, another measure of fibre
%   straightness
%
%
%   Class support for input ctfire_fibres:
%      structure array with fields x and y
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.


for single_fibre=1:length(ctfire_fibres)
    ctfire_fibre=ctfire_fibres(single_fibre);
    fibre_length = sum(((diff(ctfire_fibre.x)).^2+(diff(ctfire_fibre.y)).^2).^0.5);
    fibre_displacement = ((ctfire_fibre.x(end)-ctfire_fibre.x(1))^2+(ctfire_fibre.y(end)-ctfire_fibre.y(1))^2)^0.5;
    fibre_persistence=fibre_displacement/fibre_length;
    try % try avoids errors in alphashape generation for fibres that are 
        %totally straight
        shp =delaunayTriangulation(ctfire_fibre.x,ctfire_fibre.y);
        alpha = alphaShape(shp.Points(convexHull(shp),1),shp.Points(convexHull(shp),2));
        fibre_area = alpha.area;
        fibre_perimeter=alpha.perimeter;
        fibre_circularity = 4*pi*fibre_area/fibre_perimeter^2;
    catch
        fibre_area = 0;
        fibre_perimeter=fibre_length;
        fibre_circularity = 0;
    end
    fibre_shapes(single_fibre).length=fibre_length;
    fibre_shapes(single_fibre).displacement=fibre_displacement;
    fibre_shapes(single_fibre).persistence=fibre_persistence;
    fibre_shapes(single_fibre).area=fibre_area;
    fibre_shapes(single_fibre).perimeter=fibre_perimeter;
    fibre_shapes(single_fibre).circularity=fibre_circularity;    
end


end