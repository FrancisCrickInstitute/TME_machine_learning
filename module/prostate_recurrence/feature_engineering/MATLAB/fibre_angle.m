function [discrete_fibre_angles,end_to_end_angle_matrix,...
    end_to_end_x_derivative_matrix,end_to_end_y_derivative_matrix,...
    local_angle_matrix,local_x_derivative_matrix,...
    local_y_derivative_matrix] ... 
    = fibre_angle(ctfire_fibres, discrete_fibres, fibre_matrix)
%FIBRE_ANGLE Calculate both local and end_to_end fibre angles for each
%discrete fibre
%
%   [discrete_fibre_angles,end_to_end_angle_matrix,...
%   end_to_end_x_derivative_matrix,end_to_end_y_derivative_matrix,...
%   local_angle_matrix,local_x_derivative_matrix,...
%   local_y_derivative_matrix] = fibre_angle(ctfire_fibres, ...
%   discrete_fibres, fibre_matrix)calculates fibre angles from CTFire 
%   output, both locally and for the end_to_end fibre. Output includes a 
%   structure array of all dicrete fibre x and y derivatives (normalised to
%   be unit vectors in x and y) and angles for both local and end_to_end 
%   calculations. Equivalent matrices that includes x and y derivatives or 
%   angle for every fibre are also output.  
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
%   discrete_fibre_angles: a structure array with x and y derivatives and 
%   fibre angle both locally and end to end for each discrete fibre. 
%   Matrix output:
%   end_to_end_angle_matrix: matrix of all fibres with end to end angles 
%   for all fibre locations found in discrete_fibres 
%   end_to_end_x_derivative_matrix: matrix of all fibres with end to end x 
%   derivative values for all fibre locations found in discrete_fibres. 
%   Normalised to be unit vector in x and y 
%   end_to_end_y_derivative_matrix: matrix of all fibres with end to end y 
%   derivative values for all fibre locations found in discrete_fibres
%   Normalised to be unit vector in x and y 
%   local_angle_matrix: matrix of all fibres with local angles 
%   for all fibre locations found in discrete_fibres
%   Normalised to be unit vector in x and y 
%   local_x_derivative_matrix: matrix of all fibres with local x 
%   derivative values for all fibre locations found in discrete_fibres
%   Normalised to be unit vector in x and y 
%   local_y_derivative_matrix: matrix of all fibres with local y 
%   derivative values for all fibre locations found in discrete_fibres
%   Normalised to be unit vector in x and y 
%
%   All angles are calculated in radians. For the purposes of matrix 
%   elements where fibres overlap, the average derivative or angle is found
%   from all overlapping points and this is what is recorded in the matrix.
%   To calculate average angles, the average of the x and y derivatives is
%   calculated and the angle derived from this average.
%
%   Class support for inputs ctfire_fibres, discrete_fibres:
%      structure array with fields x and y
%   Class support for input fibre_matrix:
%      float: single, double, int: uint8, uint16, uint64
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.


all_end_to_end_x_derivative=[];
all_end_to_end_y_derivative=[];
all_end_to_end_angles=[];
all_local_x_derivative=[];
all_local_y_derivative=[];
all_local_angles=[];
all_x_location=[];
all_y_location=[];
end_to_end_angle_matrix=zeros(size(fibre_matrix))+NaN;
end_to_end_x_derivative_matrix=zeros(size(fibre_matrix))+NaN;
end_to_end_y_derivative_matrix=zeros(size(fibre_matrix))+NaN;
local_angle_matrix=zeros(size(fibre_matrix))+NaN;
local_x_derivative_matrix=zeros(size(fibre_matrix))+NaN;
local_y_derivative_matrix=zeros(size(fibre_matrix))+NaN;

for I=1:length(ctfire_fibres)

    ctfire_fibre = ctfire_fibres(I);
    discrete_fibre=discrete_fibres(I);
    %Mapping from discontinuous to continuous fibres
    ctfire_point = discrete_fibres(I).ctfire_point; 
    end_to_end_fibre_x_derivative=ctfire_fibre.x(end)-ctfire_fibre.x(1);
    end_to_end_fibre_y_derivative=ctfire_fibre.y(end)-ctfire_fibre.y(1);
    
    [end_to_end_fibre_x_derivative,end_to_end_fibre_y_derivative] = ...
        vector_normalisation(end_to_end_fibre_x_derivative,...
        end_to_end_fibre_y_derivative);
    %end_to_end_fibre_angle = atan(end_to_end_fibre_y_derivative./...
    %end_to_end_fibre_x_derivative);
    end_to_end_fibre_angle = acos(end_to_end_fibre_y_derivative);
       
    local_fibre_x_derivative=diff(ctfire_fibre.x);
    local_fibre_y_derivative=diff(ctfire_fibre.y);
    [local_fibre_x_derivative,local_fibre_y_derivative] = ...
        vector_normalisation(local_fibre_x_derivative,...
        local_fibre_y_derivative);
    %local_fibre_angle = atan(local_fibre_y_derivative./...
    %local_fibre_x_derivative);
    local_fibre_angle = acos(local_fibre_y_derivative);
    %End fibre point given same vector angle as preceding point
    local_fibre_x_derivative(end+1,1)=local_fibre_x_derivative(end);
    local_fibre_y_derivative(end+1,1)=local_fibre_y_derivative(end);
    local_fibre_angle(end+1,1)=local_fibre_angle(end);
    
   
    discrete_fibre_angles(I).local_x_derivative =  ...
        local_fibre_x_derivative(ctfire_point);
    discrete_fibre_angles(I).local_y_derivative =  ...
        local_fibre_y_derivative(ctfire_point);
    discrete_fibre_angles(I).local_fibre_angle =  ...
        local_fibre_angle(ctfire_point);
    
    discrete_fibre_angles(I).end_to_end_x_derivative =  ...
        zeros(length(ctfire_point),1)+end_to_end_fibre_x_derivative;
    discrete_fibre_angles(I).end_to_end_y_derivative =  ...
        zeros(length(ctfire_point),1)+end_to_end_fibre_y_derivative;
    discrete_fibre_angles(I).end_to_end_fibre_angle =  ...
        zeros(length(ctfire_point),1)+end_to_end_fibre_angle;
    
    linearInd = ...
        sub2ind(size(fibre_matrix),discrete_fibre.y,discrete_fibre.x);
    local_angle_matrix(linearInd) = ...
        discrete_fibre_angles(I).local_fibre_angle;
    local_x_derivative_matrix(linearInd) = ...
        discrete_fibre_angles(I).local_x_derivative;
    local_y_derivative_matrix(linearInd) = ...
        discrete_fibre_angles(I).local_y_derivative;
    end_to_end_angle_matrix(linearInd) = ...
        discrete_fibre_angles(I).end_to_end_fibre_angle;
    end_to_end_x_derivative_matrix(linearInd) = ...
        discrete_fibre_angles(I).end_to_end_x_derivative;
    end_to_end_y_derivative_matrix(linearInd) = ...
        discrete_fibre_angles(I).end_to_end_y_derivative;
    
    %Recorded so we can search for cases of overlapping fibres and take 
    %average values 
    all_end_to_end_x_derivative = [all_end_to_end_x_derivative;...
        discrete_fibre_angles(I).end_to_end_x_derivative];
    all_end_to_end_y_derivative = [all_end_to_end_y_derivative;...
        discrete_fibre_angles(I).end_to_end_y_derivative];
    all_end_to_end_angles = [all_end_to_end_angles;...
        discrete_fibre_angles(I).end_to_end_fibre_angle];
    all_local_x_derivative = [all_local_x_derivative;...
        discrete_fibre_angles(I).local_x_derivative];
    all_local_y_derivative = [all_local_y_derivative;...
        discrete_fibre_angles(I).local_y_derivative];
    all_local_angles = [all_local_angles;...
        discrete_fibre_angles(I).local_fibre_angle];
    all_x_location = [all_x_location;discrete_fibre.x];
    all_y_location = [all_y_location;discrete_fibre.y];
end
%This functionality is used to calculate the average values of x and y
%derivatives and angles where multiple fibres cross the same location.
linearInd = sub2ind(size(fibre_matrix),all_y_location,all_x_location);
[GC,GR] = groupcounts(linearInd);
find_overlaps=find(GC>1);
for J=1:length(find_overlaps)
    mean_end_to_end_x_derivative = mean(all_end_to_end_x_derivative(...
        find(linearInd==GR(find_overlaps(J)))));
    mean_end_to_end_y_derivative = mean(all_end_to_end_y_derivative(...
        find(linearInd==GR(find_overlaps(J)))));
    mean_local_x_derivative=mean(all_local_x_derivative(...
        find(linearInd==GR(find_overlaps(J)))));
    mean_local_y_derivative=mean(all_local_y_derivative(...
        find(linearInd==GR(find_overlaps(J)))));    
    end_to_end_x_derivative_matrix(GR(find_overlaps(J))) = ...
        mean_end_to_end_x_derivative;
    end_to_end_y_derivative_matrix(GR(find_overlaps(J))) = ...
        mean_end_to_end_y_derivative;
    local_x_derivative_matrix(GR(find_overlaps(J))) = ...
        mean_local_x_derivative;
    local_y_derivative_matrix(GR(find_overlaps(J))) = ...
        mean_local_y_derivative;
    
    %Calculate the angles from the average x and y vectors rather than the 
    % angle itself (to account for periodicity in angle).
    [mean_end_to_end_x_derivative,mean_end_to_end_y_derivative] = ...
        vector_normalisation(mean_end_to_end_x_derivative,...
        mean_end_to_end_y_derivative);
    %end_to_end_fibre_angle = atan(mean_end_to_end_y_derivative./...
    %mean_end_to_end_x_derivative);
    end_to_end_fibre_angle = acos(mean_end_to_end_y_derivative);
    end_to_end_angle_matrix(GR(find_overlaps(J))) = end_to_end_fibre_angle;
    [mean_local_x_derivative,mean_local_y_derivative] = ...
        vector_normalisation(mean_local_x_derivative,...
        mean_local_y_derivative);
%   local_fibre_angle = atan(mean_local_y_derivative./...
%   mean_local_x_derivative);
    local_fibre_angle = acos(mean_local_y_derivative);
    local_angle_matrix(GR(find_overlaps(J))) = local_fibre_angle;
   
end

end

