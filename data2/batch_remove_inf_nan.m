function [outputArg1,outputArg2] = batch_remove_inf_nan(file_dir,names)
%BATCH_ 此处显示有关此函数的摘要
%   此处显示详细说明
for ii =  length(names) : -1:1
    file_path = strcat(file_dir, names(ii).name);
    remove_inf_nan(file_path);
end
end

