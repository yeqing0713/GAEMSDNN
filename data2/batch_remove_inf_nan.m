function [outputArg1,outputArg2] = batch_remove_inf_nan(file_dir,names)
%BATCH_ �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
for ii =  length(names) : -1:1
    file_path = strcat(file_dir, names(ii).name);
    remove_inf_nan(file_path);
end
end

