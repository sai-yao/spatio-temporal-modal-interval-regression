function [p_lower,p_upper,p_lower_value,p_upper_value] = cmie_b(dataset,x,y,alpha,bandwidth_x,bandwidth_y)
%CMIE_B 此处显示有关此函数的摘要
%   此处显示详细说明
ds = dataset;
n = length(ds);
Modal_Interval_Size = Inf;
qua_val = zeros(n+1,2);%quantile+value
for i = 2:n+1
    [qua_val(i,1),qua_val(i,2)] = cdfe_b(ds,x, y, ds(i-1,3),bandwidth_x, bandwidth_y);
end
qua_val(1,2)=min(ds(:,3));

for i = 1:n+1
    if qua_val(i,1)>1-alpha
        break;
    end
    for j = i:n+1
        if qua_val(j,1)-qua_val(i,1)>=alpha
            Interval_Size = qua_val(j,2) - qua_val(i,2);
            if Modal_Interval_Size > Interval_Size
                Modal_Interval_Size = Interval_Size;
                p_lower = qua_val(i,1);
                p_upper = qua_val(j,1);
                p_lower_value = qua_val(i,2);
                p_upper_value = qua_val(j,2);
            end
            break;
        end
    end
end

end

