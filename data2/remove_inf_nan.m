function [outputArg1,outputArg2] = remove_inf_nan(path)
%REMOVE_INF_NAN 此处显示有关此函数的摘要
%   此处显示详细说明
 aa = csvread(path);
 aa(isnan(aa)) = 0;
 aa(isinf(aa)) = 0;
 csvwrite(path, aa);
end

