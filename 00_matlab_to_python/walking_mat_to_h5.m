% Clear workspace
close all; clc; clear all;

% Load Calgary data
dat_name = 'walking';
mat_name = [dat_name '.mat'];
h5_name = [dat_name '_meta.h5'];
load(mat_name);

% Check consistency
err_msg = 'Number of entries are not consistent!';
assert( height(meta) == length(markerdata) && ...
        height(meta) == length(jointdata) && ...
        height(meta) == length(events), err_msg);

% Loop over subjects
for i = 1:length(events)    
    % Extract data of current subject
    s_id = ['s' num2str(i)];
    cur_events = events{i,1};
    cur_angles = jointdata{i,1};
    cur_markers = markerdata{i,1};
    cur_meta = meta(i,:);
    
    % Remove toe markers if they exist
    if length(fieldnames(cur_markers)) == 30
        cur_markers = rmfield(cur_markers,'R_toe');
        cur_markers = rmfield(cur_markers,'L_toe');
    end
    
    % Add markerdata of current subject
    f_names = fieldnames(cur_markers);
    for f_id = 1:length(f_names)
        d = cur_markers.(f_names{f_id, 1});
        pth = ['/' s_id '/markers/' f_names{f_id, 1}];
        % Transpose size and array to change from F- to C-style memory saving
        sz = size(d);
        h5create(h5_name, pth, [sz(2) sz(1)])
        h5write(h5_name, pth, d')
    end

    % Add jointangles of current subject
    f_names = fieldnames(cur_angles);
    for f_id = 1:length(f_names)
        d = cur_angles.(f_names{f_id, 1});
        pth = ['/' s_id '/angles/' f_names{f_id, 1}];
        % Transpose size and array to change from F- to C-style memory saving
        sz = size(d);
        h5create(h5_name, pth, [sz(2) sz(1)])
        h5write(h5_name, pth, d')
    end

    % Add events of current subject
    d_ev = cur_events;
    pth = ['/' s_id '/events'];
    % Transpose size and array to change from F- to C-style memory saving
    sz = size(d_ev);
    h5create(h5_name, pth, [sz(2) sz(1)])
    if is2dDataArray(d_ev)
        h5write(h5_name, pth, d_ev')
    end
    
    % Add meta variables of current subject
    d = cur_meta;
    d_cell = table2cell(d);  
    d_new = {d_cell{1,1}, 100, d_cell{1,3}, d_cell{1,4}, d_cell{1,5}};
    
    % Assigns 0 for male subjects and 1 for female subjects
    if isequal(d_cell(1,2),{'Male'})
        d_new(1,2) = {0.0};
    end
    if isequal(d_cell(1,2),{'Female'})
        d_new(1,2) = {1.0};
    end
    
    % Assigns -1 if meta data is missing
    for j = 1:length(d_new)
        if isequal(d_new(1,j), {'NA'})
            d_new(1,j) = {-1.0};
        end
    end
    
    d_mat = cell2mat(d_new);   
    pth = ['/' s_id '/meta'];
    
    sz2 = size(d_mat);
    h5create(h5_name, pth, [sz2(2) sz2(1)])
    h5write(h5_name, pth, d_mat')
    
    % Updates user on progress
    disp(strcat(num2str(i),'/',num2str(length(events))))
    
end

