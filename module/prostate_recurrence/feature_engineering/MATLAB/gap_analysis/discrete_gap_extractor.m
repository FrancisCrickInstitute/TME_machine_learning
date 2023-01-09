function discrete_gap_labels = discrete_gap_extractor(ecm_mask,tissue_mask,image)


tissue_boundary = bwmorph(tissue_mask,'remove');
SE = strel('disk',10,0);
ecm_close=imclose(ecm_mask,SE);
candidate_objects = logical(~ecm_close.*tissue_mask);
object_labels = bwlabel(candidate_objects,8);

unique_objects = unique(object_labels(:));
unique_objects=unique_objects(unique_objects>0);
boundary_objects=zeros(size(image));
for object_index=1:length(unique_objects)
    single_object = zeros(size(image));
    index=find(object_labels==unique_objects(object_index));
    single_object(index)=1;
    boundary_overlap = single_object.*tissue_boundary;
    if max(boundary_overlap(:))==1
        boundary_objects(index)=1;
        object_labels(index)=0;
    end
end
eroded_boundary = imerode(boundary_objects,SE);
boundary_label = bwlabel(eroded_boundary,8);
boundary_index=find(boundary_label);
boundary_label(boundary_index)=boundary_label(boundary_index)+max(object_labels(:));
object_labels=max(object_labels,boundary_label);

raw_gaps = logical(~ecm_mask.*tissue_mask);
discrete_gap_labels = bwlabel(raw_gaps,8);
unique_discrete_gap_labels=unique(discrete_gap_labels);
unique_discrete_gap_labels=unique_discrete_gap_labels(unique_discrete_gap_labels>0);
max_object_label = max(discrete_gap_labels(:));
for discrete_gap_index=1:length(unique_discrete_gap_labels)
    single_raw_gap = zeros(size(image));
    single_raw_gap_index=find(discrete_gap_labels==unique_discrete_gap_labels(discrete_gap_index));
    single_raw_gap(single_raw_gap_index)=1;
    raw_overlap = single_raw_gap.*object_labels;
    unique_raw_overlap = unique(raw_overlap(:));
    unique_raw_overlap=unique_raw_overlap(unique_raw_overlap>0);
    if length(unique_raw_overlap)>1
        clear all_distances
        discrete_gap_labels(single_raw_gap_index)=0;
        for overlap_object=1:length(unique_raw_overlap)
            index_object=find(object_labels==unique_raw_overlap(overlap_object));
            single_object=zeros(size(image));
            single_object(index_object)=1;
            distance_object=bwdist(single_object);
            all_distances(:,overlap_object) = distance_object(single_raw_gap_index);
        end
        [M,object_index] = min(all_distances');
        for overlap_object=1:length(unique_raw_overlap)
            max_object_label = max_object_label +1;
            index_min_distance = find(object_index==overlap_object);
            input_array = logical(zeros(size(image)));
            input_array(single_raw_gap_index(index_min_distance))=1;
            input_array = bwareafilt(input_array,1);
            connected_object_index = find(input_array);
            discrete_gap_labels(connected_object_index )=max_object_label; 
        end
    end
end

unique_discrete_gap_labels=unique(discrete_gap_labels);
unique_discrete_gap_labels=unique_discrete_gap_labels(unique_discrete_gap_labels>0);
for relabel_index=1:length(unique_discrete_gap_labels)
    object_index=find(discrete_gap_labels==unique_discrete_gap_labels(relabel_index));
    discrete_gap_labels(object_index)=relabel_index;
end
unique_discrete_gap_labels=unique(discrete_gap_labels);
unique_discrete_gap_labels=unique_discrete_gap_labels(unique_discrete_gap_labels>0);

%Add more aggressive filter for failed large cases
for object_index=1:length(unique_discrete_gap_labels)
    object_size(object_index) = length(find(discrete_gap_labels==unique_discrete_gap_labels(object_index)));
end
large_index=find(object_size>=200000);
if isempty(large_index)==0
    large_label_index = unique_discrete_gap_labels(large_index);

    SE = strel('disk',30,0);
    ecm_close=imclose(ecm_mask,SE);
    candidate_objects = logical(~ecm_close.*tissue_mask);
    object_labels = bwlabel(candidate_objects,8);

    unique_objects = unique(object_labels(:));
    unique_objects=unique_objects(unique_objects>0);
    boundary_objects=zeros(size(image));
    for object_index=1:length(unique_objects)
        single_object = zeros(size(image));
        index=find(object_labels==unique_objects(object_index));
        single_object(index)=1;
        boundary_overlap = single_object.*tissue_boundary;
        if max(boundary_overlap(:))==1
            boundary_objects(index)=1;
            object_labels(index)=0;
        end
    end
    eroded_boundary = imerode(boundary_objects,SE);
    boundary_label = bwlabel(eroded_boundary,8);
    boundary_index=find(boundary_label);
    boundary_label(boundary_index)=boundary_label(boundary_index)+max(object_labels(:));
    object_labels=max(object_labels,boundary_label);


    for discrete_gap_index=1:length(large_label_index)
        single_raw_gap = zeros(size(image));
        single_raw_gap_index=find(discrete_gap_labels==large_label_index(discrete_gap_index));
        single_raw_gap(single_raw_gap_index)=1;
        raw_overlap = single_raw_gap.*object_labels;
        unique_raw_overlap = unique(raw_overlap(:));
        unique_raw_overlap=unique_raw_overlap(unique_raw_overlap>0);
        if length(unique_raw_overlap)>1
            clear all_distances
            discrete_gap_labels(single_raw_gap_index)=0;
            for overlap_object=1:length(unique_raw_overlap)
                index_object=find(object_labels==unique_raw_overlap(overlap_object));
                single_object=zeros(size(image));
                single_object(index_object)=1;
                distance_object=bwdist(single_object);
                all_distances(:,overlap_object) = distance_object(single_raw_gap_index);
            end
            [M,object_index] = min(all_distances');
            for overlap_object=1:length(unique_raw_overlap)
                max_object_label = max_object_label +1;
                index_min_distance = find(object_index==overlap_object);
                input_array = logical(zeros(size(image)));
                input_array(single_raw_gap_index(index_min_distance))=1;
                input_array = bwareafilt(input_array,1);
                connected_object_index = find(input_array);
                discrete_gap_labels(connected_object_index )=max_object_label; 
            end
        end
    end

    unique_discrete_gap_labels=unique(discrete_gap_labels);
    unique_discrete_gap_labels=unique_discrete_gap_labels(unique_discrete_gap_labels>0);
    for relabel_index=1:length(unique_discrete_gap_labels)
        object_index=find(discrete_gap_labels==unique_discrete_gap_labels(relabel_index));
        discrete_gap_labels(object_index)=relabel_index;
    end
    unique_discrete_gap_labels=unique(discrete_gap_labels);
    unique_discrete_gap_labels=unique_discrete_gap_labels(unique_discrete_gap_labels>0);


end
