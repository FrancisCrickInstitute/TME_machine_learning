function [discrete_fibre_curvature,curvature_matrix,curvature_continuum]...
    = fibre_curvature(ctfire_fibres, discrete_fibres, fibre_matrix)
%FIBRE_CURVATURE Calculate curvature for each discrete fibre
%
%   [discrete_fibre_curvature,curvature_matrix] = 
%   fibre_curvature(ctfire_fibres, discrete_fibres, fibre_matrix)
%   calculates fibre curvature from CTFire output. Output includes a 
%   structure array of all curvatures and an equivalent
%   matrix of all curvature values. Where fibres overlap on the matrix the 
%   average curvature is calculated. Curvature of zero is adjusted to 
%   10^(-50) to avoid issues with the skewness and kurtosis functions. 
%
%   Input:
%   ctfire_fibres: Structure array of discrete CTFire fibres. Used to 
%   calculate both local and end to end fibre angles.  
%   discrete_fibres: Interpolated discrete fibres to remove missing points
%   from CTFire. These are the fibres used to impose angle information.
%   fibre_matrix: Matrix of all discrete fibres. Used here only for image 
%   size information 
%
%   Output:
%   discrete_fibre_curvature: a structure array with curvature for each 
%   discrete fibre. 
%   curvature_matrix: matrix of all curvatures for all fibres. Where fibres
%   overlap, the average curvature is taken
%
%   Class support for inputs ctfire_fibres, discrete_fibres:
%      structure array with fields x and y
%   Class support for input fibre_matrix:
%      float: single, double, int: uint8, uint16, uint64
%
%   Curvature is calculated using the function LineCurvature2D by
%   Dirk-Jan Kroon (2021). 2D Line Curvature and Normals
%   (https://www.mathworks.com/matlabcentral/...
%   fileexchange/32696-2d-line-curvature-and-normals), 
%   MATLAB Central File Exchange. Retrieved May 11, 2021.
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.

curvature_matrix=zeros(size(fibre_matrix))+NaN;
all_curvature=[];
all_x_location=[];
all_y_location=[];
for I=1:length(ctfire_fibres)
    ctfire_fibre = ctfire_fibres(I);
    discrete_fibre=discrete_fibres(I);
    %Mapping from discontinuous to continuous fibres
    ctfire_point = discrete_fibres(I).ctfire_point; 
    Vertices=[ctfire_fibre.x,ctfire_fibre.y];
    curvature=abs(LineCurvature2D(Vertices));
    discrete_fibre_curvature(I).curvature =  curvature(ctfire_point);
    %If fibre contains only two points then will produce NaN values
    if max(isnan(curvature))==0
        linearInd = sub2ind(...
            size(fibre_matrix),...
            discrete_fibre.y,...
            discrete_fibre.x...
            );
        curvature_matrix(linearInd)=discrete_fibre_curvature(I).curvature;
    
    %   Recorded so we can search for cases of overlapping fibres and take 
    %   average values 
        all_curvature=[all_curvature;curvature(ctfire_point)];
        all_x_location=[all_x_location;discrete_fibre.x];
        all_y_location=[all_y_location;discrete_fibre.y];
    end
end
%This functionality is used to calculate the average values of x and y
%derivatives and angles where multiple fibres cross the same location.
linearInd = sub2ind(size(fibre_matrix),all_y_location,all_x_location);
[GC,GR] = groupcounts(linearInd);
find_overlaps=find(GC>1);
for J=1:length(find_overlaps)
    mean_curvature = ...
        mean(all_curvature(linearInd==GR(find_overlaps(J))));
        %mean(all_curvature(find(linearInd==GR(find_overlaps(J)))));
    curvature_matrix(GR(find_overlaps(J))) = mean_curvature;
end
%Use spring metaphor method for inpainting
curvature_continuum  = inpaint_nans(curvature_matrix,4); 
curvature_continuum(curvature_continuum==0)=10^-50;
end

