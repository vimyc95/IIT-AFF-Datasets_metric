function F_wb_non_rank = evaluate_Fwb_non_rank(path_predited, path_gt)

% affordances index
aff_start=2;   % ignore {background} label
aff_end=10;   % change based on the dataset 

% get all files
list_predicted = getAllFiles(path_predited);   % get all files in current folder
list_gt = getAllFiles(path_gt);
list_predicted = sort(cellstr(list_predicted));
list_gt = sort(cellstr(list_gt)); % make the same style
c1 = length(list_predicted)
c2 = length(list_gt)
assert(length(list_predicted)==length(list_gt)); % test length
num_of_files = length(list_gt);

F_wb_aff = nan(num_of_files,1);
F_wb_non_rank = [];

for aff_id = aff_start:aff_end  % from 2 --> final_aff_id
    for i=1:num_of_files
        
        fprintf('------------------------------------------------\n');
        fprintf('affordance id=%d, image i=%d \n', aff_id, i);
        fprintf('current pred: %s\n', list_predicted{i});
        fprintf('current grth: %s\n', list_gt{i});
        
        %%read image      
        pred_im = imread(list_predicted{i}); 
        gt_im = imread(list_gt{i});

        fprintf('size pred_im: %d \n', size(pred_im));
        fprintf('size gt_im  : %d \n', size(gt_im));
        
        pred_im = pred_im(:,:,1);
        gt_im = gt_im(:,:,1);
       
        targetID = aff_id - 1; %labels are zero-indexed so we minus 1
        
        % only get current affordance
        pred_aff = pred_im == targetID;
        gt_aff = gt_im == targetID;
        
        if sum(gt_aff(:)) > 0 % only compute if the affordance has ground truth
            F_wb_aff(i,1) = WFb(double(pred_aff), gt_aff);  % call WFb function
        else
            %fprintf('no ground truth at i=%d \n', i);
        end
        
    end
    fprintf('Averaged F_wb for affordance id=%d is: %f \n', aff_id-1, nanmean(F_wb_aff));
    F_wb_non_rank = [F_wb_non_rank; nanmean(F_wb_aff)];
%     fprintf('Averaged F_wb for affordance id=%d is: %f \n', aff_id-1, mean(F_wb_aff));
%     F_wb_non_rank = [F_wb_non_rank; mean(F_wb_aff)];
    
end
ans_m = nanmean(F_wb_non_rank)

end