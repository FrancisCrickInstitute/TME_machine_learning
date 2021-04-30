function [fibre_skeleton,fibre_branchpoints,fibre_endpoints,fibre_disconnected_branches] = fibre_skeletonisation(fibre_matrix)
%FIBRE_SKELETONISATION Transform fibre network to fibre skeleton to allow
%graph theory based analysis of fibre matrix 
%
%   [fibre_skeleton,fibre_branchpoints,...
%    fibre_endpoints,fibre_disconnected_branches] 
%   = fibre_skeletonisation(fibre_matrix) skeletonises fibre_matrix and
%   extracts a barnchpoint map, endpoint map and disconnected branches map.
%
%   Input:
%   fibre_matrix: matrix of labelled discret fibrese. 
%
%   Output:
%   fibre_skeleton: binary matrix of skeletonised fibre network
%   fibre_branchpoints: binary matrix of branchpoints
%   fibre_endpoints: binary matrix of endpoints
%   fibre_disconnected_branches: binary matrix of fibres disconnected by 
%   removal of branchpoints


%   The below code for brnchpoints and endpoints appears to work better than
%   MATLAB's inbuilt functionality for 8 connected neighbourhoods.
%
%   Class support for input fibre_matrix:
%      float: single, double, int: uint8, uint16, uint64
%   
%
%   This work is licensed under a Creative Commons Attribution 4.0 
%   International License.

fibre_matrix = logical(fibre_matrix);
fibre_skeleton = bwmorph(fibre_matrix,'skel',Inf);
fibre_branchpoints=bwlookup(fibre_skeleton,  makelut(@(x) sum(x(:))>=4 & x(5)==1,3));
endpoint_fcn = @(nhood) (nhood(2,2) ~= 0) && (sum(nhood(:)) == 2);
endpoint_lut = makelut(endpoint_fcn, 3);
fibre_endpoints = applylut(fibre_skeleton, endpoint_lut);
fibre_disconnected_branches=fibre_skeleton-fibre_branchpoints;

