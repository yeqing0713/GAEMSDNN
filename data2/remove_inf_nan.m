function [outputArg1,outputArg2] = remove_inf_nan(path)
%REMOVE_INF_NAN �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
 aa = csvread(path);
 aa(isnan(aa)) = 0;
 aa(isinf(aa)) = 0;
 csvwrite(path, aa);
end

