function [quantile, value,w1,w2] = cdfe_b(dataset, x, y, z, bandwidth_x, bandwidth_y)
%ccdfe_b conditional cumulative distribution functuon estimation for bivariate
%dataset: dataset
%x, y, z: value of x, y and z
%bandwidth: bandwidth of kernel function
ds = dataset;
n = length(ds);
weight = zeros(n,2);
for i = 1:n
    weight(i,1)=exp( -abs((ds(i,1) - x)/bandwidth_x)^2 / 2) * exp( -abs((ds(i,2) - y)/bandwidth_y)^2 / 2);
    weight(i,2)=ds(i,3);
end
total_weight = 0;
for i = 1 : n
    total_weight = total_weight + weight(i,1);
end
accumulate_weight = 0;
weight = sortrows(weight,2);
for i = 1 : n
    if weight(i,2) <= z
        accumulate_weight = accumulate_weight + weight(i,1);
    else
        break;
    end
end

quantile = accumulate_weight / total_weight;
value = z;
w1=accumulate_weight;
w2=total_weight;
end

