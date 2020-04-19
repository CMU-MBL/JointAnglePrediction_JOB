% Load Calgary data
dat_name = 'walking';
mat_name = [dat_name '.mat'];
load(mat_name);

% Checks for unique values
x = [];
for i=1:height(meta)
    newRow = [];
    for j=3:4
        tmp = meta{i, j};
        if ~isnumeric(tmp{:})
            newRow = [newRow, 0];
        else
            newRow = [newRow, tmp{:}];
        end
    end
    newRow = [newRow, meta{i,5}];
    x = [x; newRow];
end
tmp = unique(x, 'rows');

disp(['Number of unique subjects reported in the dataset: ', num2str(height(meta))])
disp(['Found ', num2str(length(tmp)), ' unique rows in the meta data.'])