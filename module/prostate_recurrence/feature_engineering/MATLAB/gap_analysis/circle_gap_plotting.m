function circle_gap_plotting(label_matrix,input_gap_matrix,filename)
%CIRCLE_GAP_PLOTTING plots the gap filling circles with random colour
%alongside the fibres in white.
% pixel and the nearest fibre for a logical input matrix of fibres and
% produces related output data.
%
%    circle_gap_plotting(label_matrix,filename) takes the index notated 
%    label matrix of all fitted circles and converts to rgb with each 
%    circle shown in a different, random colour. Alongside this the fibres
%    are shown in white. The resulting rgb image is then saved as a tiff.
%
%   Input:
%   label_matrix: Pixels labelled by unsigned integer values determining 
%   correspondence to a given circle.
%   filename: The directory, filename and file extension used to save the 
%   resulting rgb image.
%
%
%   Class support for input label_matrix:
%      float: single, double
%   Class support for input filename:
%      string
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.

cm=rand(length(unique(label_matrix))+1,3);
cm(1,:)=0;
RGB = ind2rgb(label_matrix+1,cm);
RGB(:,:,1)=max(RGB(:,:,1),input_gap_matrix);
RGB(:,:,2)=max(RGB(:,:,2),input_gap_matrix);
RGB(:,:,3)=max(RGB(:,:,3),input_gap_matrix);
imwrite(RGB,filename)
end