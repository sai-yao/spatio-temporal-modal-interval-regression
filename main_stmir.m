%% Data
X = X; % spatial covariate vector X
Y = Y; % spatial covariate vector Y
T = T; % temporal covariate vector T
Z = Z; % response vector

test_X = test_X;
test_Y = test_Y;
test_T = test_T;
test_Z = test_Z;


%% Parameter
alpha = 0.8;          % coverage level
J = 5;                % number of polynomial pieces for X
K = 5;                % number of polynomial pieces for Y
start_value_x = 0;    % covariate X left endpoint
start_value_y = 0;    % covariate Y left endpoint
end_value_x = 10;     % covariate X right endpoint
end_value_y = 10;     % covariate Y right endpoint
d1 = 2;               % degree of temporal polynomial
d2 = 3;               % degree of spatial polynomial
rho = 1;              % smoothness of spline
lambda = 0.001;       % smoothing parameter
iter = 500;          % number of ADMM iterations
eta = 20;             % mCWC penalty strength parameter
n_kde = 100;         % maximum number of data points used for KDE-based quantile level estimation
n_time_groups = 4;   % number of time groups used for KDE-based estimation

map_width = 'turbo';  % colormap for interval width
map_upper = 'turbo';  % colormap for upper bound
map_lower = 'turbo';  % colormap for lower bound

delta_x = (end_value_x - start_value_x) / J;
delta_y = (end_value_y - start_value_y) / K;
spline_knots_X = start_value_x:delta_x:end_value_x;
spline_knots_Y = start_value_y:delta_y:end_value_y;

Z_range = max(Z) - min(Z);
if Z_range > 1
    Z_normalize = Z_range;
else
    Z_normalize = 1;
end

d_data = Z / Z_normalize;

time_edges = linspace(0, 1, n_time_groups + 1);
time_midpoints = (time_edges(1:end-1) + time_edges(2:end)) / 2;

T_group_index = discretize(T, time_edges);
T_group_index(T == max(T)) = n_time_groups;
T_label = time_midpoints(T_group_index)';


sort_data = sortrows([T_label, T, X, Y, d_data], [1, 2, 3, 4]);
sort_TXY = sortrows([T_label, T, X, Y], [1, 2, 3, 4]);

[u, ~, idx] = unique(T_label);
Time_label_num = numel(u);
Time_label_counts = accumarray(idx, 1);

n_s = length(T);
L = 2;

bandwidth_x = zeros(Time_label_num, 1);
bandwidth_y = zeros(Time_label_num, 1);

for t0 = 1:Time_label_num

    XY_selected = sort_TXY(sort_TXY(:,1) == u(t0), 3:4);

    if size(XY_selected, 1) >= n_kde
        rand_rows = randperm(size(XY_selected, 1), n_kde);
        kde_matrix = XY_selected(rand_rows, :);
        [~, ~, bandwidth] = ksdensity(kde_matrix, 'Bandwidth', 'normal-approx');
    else
        [~, ~, bandwidth] = ksdensity(XY_selected, 'Bandwidth', 'normal-approx');
    end

    bandwidth_x(t0) = bandwidth(1);
    bandwidth_y(t0) = bandwidth(2);
end


%% Generate matrices
%%% Weight Vector: w %%%
w_cell = cell(Time_label_num, 1);

for t0 = 1:Time_label_num

    w0 = zeros(Time_label_counts(t0), 1);
    XY_selected = sort_TXY(sort_TXY(:,1) == u(t0), 3:4);

    for i = 1:Time_label_counts(t0)
        w0(i,1) = nthroot(ksdensity(XY_selected, XY_selected(i,:), 'Bandwidth', 'normal-approx'), 5);
    end

    w_cell{t0} = w0;
end

w = vertcat(w_cell{:});


%%% Quantile Level Vector: p %%%
p_upper_cell = cell(Time_label_num, 1);
p_lower_cell = cell(Time_label_num, 1);

n_done = 0;

for t0 = 1:Time_label_num

    XYZ_selected = sort_data(sort_data(:,1) == u(t0), 3:5);

    p_upper_0 = zeros(Time_label_counts(t0), 1);
    p_lower_0 = zeros(Time_label_counts(t0), 1);

    SORT = sortrows(XYZ_selected, 3);

    for i = 1:Time_label_counts(t0)

        n_done = n_done + 1;

        if mod(n_done, 100) == 0 || n_done == n_s
            fprintf('Quantile level estimation: %d / %d\n', n_done, n_s);
        end

        if size(XYZ_selected, 1) > n_kde
            rand_rows = randperm(size(XYZ_selected, 1), n_kde);
            kde_matrix = SORT(rand_rows, :);

            [p_lower_0(i), p_upper_0(i)] = cmie_b(kde_matrix, ...
                XYZ_selected(i,1), XYZ_selected(i,2), ...
                alpha, bandwidth_x(t0), bandwidth_y(t0));
        else
            [p_lower_0(i), p_upper_0(i)] = cmie_b(SORT, ...
                XYZ_selected(i,1), XYZ_selected(i,2), ...
                alpha, bandwidth_x(t0), bandwidth_y(t0));
        end
    end

    p_lower_0(p_lower_0 < 0.01) = 0.01;
    p_upper_0(p_upper_0 > 0.99) = 0.99;

    p_upper_cell{t0} = p_upper_0;
    p_lower_cell{t0} = p_lower_0;
end

p_upper = vertcat(p_upper_cell{:});
p_lower = vertcat(p_lower_cell{:});

%%% Observations Matrix: A %%%

n_coef_piece = (d1 + 1) * (d2 + 1) * (d2 + 2) / 2;

iii = ones(L * n_s * n_coef_piece, 1);
jjj = ones(L * n_s * n_coef_piece, 1);
A_s = zeros(L * n_s * n_coef_piece, 1);

ii = 1;

for kk = 1:L

    n_all = 0;

    for t0 = 1:Time_label_num

        % Columns:
        % TXY_selected(:,1): original continuous T
        % TXY_selected(:,2): X
        % TXY_selected(:,3): Y
        TXY_selected = sort_data(sort_data(:,1) == u(t0), 2:4);

        n_t0 = size(TXY_selected, 1);

        for i = 1:Time_label_counts(t0)

            %%%Find the subdomain in the X-direction
            for j = 1:J
                if TXY_selected(i,2) <= spline_knots_X(j+1)
                    bk_X = j;
                    break
                end
            end

            %%%Find the subdomain in the Y-direction
            for k = 1:K
                if TXY_selected(i,3) <= spline_knots_Y(k+1)
                    bk_Y = k;
                    break
                end
            end

            bk = (bk_Y - 1) * J + bk_X;

            tau_x = (TXY_selected(i,2) - spline_knots_X(bk_X)) / delta_x;
            tau_y = (TXY_selected(i,3) - spline_knots_Y(bk_Y)) / delta_y;

            for dd_t = 0:d1
                for dd_y = 0:d2
                    for dd_x = 0:(d2 - dd_y)

                        coef_idx_local = ...
                            dd_t * (d2 + 1) * (d2 + 2) / 2 + ...
                            dd_y * (d2 + (3 - dd_y) / 2) + ...
                            dd_x + 1;

                        iii(ii + coef_idx_local - 1) = ...
                            (kk - 1) * n_s + n_all + i;

                        jjj(ii + coef_idx_local - 1) = ...
                            (kk - 1) * J * K * n_coef_piece + ...
                            (bk - 1) * n_coef_piece + ...
                            coef_idx_local;

                        A_s(ii + coef_idx_local - 1) = ...
                            TXY_selected(i,1)^dd_t * ...
                            tau_x^dd_x * ...
                            tau_y^dd_y;

                    end
                end
            end

            ii = ii + n_coef_piece;
        end

        n_all = n_all + n_t0;
    end
end

A = sparse(iii, jjj, A_s, L * n_s, L * J * K * n_coef_piece);


%%%Regularzation Term Matrix: Q%%%
S_1=zeros((d1+1)*(d2+1)*(d2+2)/2,(d1+1)*(d2+1)*(d2+2)/2);
for h=0:d1
    for m=0:d2
        for g=0:d2-m
            for l=0:d1
                for v=0:d2
                    for r=0:d2-v
                        if (g>=2 && m<=d2-2) && (r>=2 && v<=d2-2)
                            S_1(l*(d2+1)*(d2+2)/2+v*(d2+(3-v)/2)+r+1,h*(d2+1)*(d2+2)/2+m*(d2+(3-m)/2)+g+1)=(g*(g-1)*r*(r-1)*delta_y)/(delta_x^3*(g+r-3)*(m+v+1)*(h+l+1));
                        end
                    end
                end
            end
        end
    end
end
S_1=(S_1+S_1')/2;

S_2=zeros((d1+1)*(d2+1)*(d2+2)/2,(d1+1)*(d2+1)*(d2+2)/2);
for h=0:d1
    for m=0:d2
        for g=0:d2-m
            for l=0:d1
                for v=0:d2
                    for r=0:d2-v
                        if (m>=2 && g<=d2-2) && (v>=2 && r<=d2-2)
                            S_2(l*(d2+1)*(d2+2)/2+v*(d2+(3-v)/2)+r+1,h*(d2+1)*(d2+2)/2+m*(d2+(3-m)/2)+g+1)=(m*(m-1)*v*(v-1)*delta_x)/(delta_y^3*(m+v-3)*(g+r+1)*(h+l+1));
                        end
                    end
                end
            end
        end
    end
end
S_2=(S_2+S_2')/2;

S_3=zeros((d1+1)*(d2+1)*(d2+2)/2,(d1+1)*(d2+1)*(d2+2)/2);
for h=0:d1
    for m=0:d2
        for g=0:d2-m
            for l=0:d1
                for v=0:d2
                    for r=0:d2-v
                        if (m>=1 && m<=d2-1 && g>=1) && (v>=1 && v<=d2-1 && r>=1)
                            S_3(l*(d2+1)*(d2+2)/2+v*(d2+(3-v)/2)+r+1,h*(d2+1)*(d2+2)/2+m*(d2+(3-m)/2)+g+1)=(g*m*r*v)/(delta_y*delta_x*(m+v-1)*(g+r-1)*(h+l+1));
                        end
                    end
                end
            end
        end
    end
end
S_3=(S_3+S_3')/2;

S=S_1+S_2+2*S_3;


iii=zeros(L*J*K*(d1+1)^2*(d2+1)^2*(d2+2)^2/4,1);
jjj=zeros(L*J*K*(d1+1)^2*(d2+1)^2*(d2+2)^2/4,1);
Q_s=zeros(L*J*K*(d1+1)^2*(d2+1)^2*(d2+2)^2/4,1);
for kk=1:L
    for k=1:K
        for j=1:J
            ii=(k-1)*J+j;
            for h=0:d1
                for m=0:d2
                    for g=0:d2-m
                        for l=0:d1
                            for v=0:d2
                                for r=0:d2-v
                                    iii((kk-1)*J*K*((d1+1)^2)*((d2+1)^2)*((d2+2)^2)/4 + (ii-1)*((d1+1)^2)*((d2+1)^2)*((d2+2)^2)/4 + (h*(d2+1)*(d2+2)/2+m*(d2+(3-m)/2)+g)*((d1+1)*(d2+1)*(d2+2)/2) + l*(d2+1)*(d2+2)/2 + v*(d2+(3-v)/2)+r+1) = (kk-1)*J*K*(d1+1)*(d2+1)*(d2+2)/2 + (ii-1)*(d1+1)*(d2+1)*(d2+2)/2 + l*(d2+1)*(d2+2)/2 + v*(d2+(3-v)/2)+r+1;
                                    jjj((kk-1)*J*K*((d1+1)^2)*((d2+1)^2)*((d2+2)^2)/4 + (ii-1)*((d1+1)^2)*((d2+1)^2)*((d2+2)^2)/4 + (h*(d2+1)*(d2+2)/2+m*(d2+(3-m)/2)+g)*((d1+1)*(d2+1)*(d2+2)/2) + l*(d2+1)*(d2+2)/2 + v*(d2+(3-v)/2)+r+1) = (kk-1)*J*K*(d1+1)*(d2+1)*(d2+2)/2 + (ii-1)*(d1+1)*(d2+1)*(d2+2)/2 + h*(d2+1)*(d2+2)/2 + m*(d2+(3-m)/2)+g+1;
                                    Q_s((kk-1)*J*K*((d1+1)^2)*((d2+1)^2)*((d2+2)^2)/4 + (ii-1)*((d1+1)^2)*((d2+1)^2)*((d2+2)^2)/4 + (h*(d2+1)*(d2+2)/2+m*(d2+(3-m)/2)+g)*((d1+1)*(d2+1)*(d2+2)/2) + l*(d2+1)*(d2+2)/2 + v*(d2+(3-v)/2)+r+1) = S(l*(d2+1)*(d2+2)/2+v*(d2+(3-v)/2)+r+1,h*(d2+1)*(d2+2)/2+m*(d2+(3-m)/2)+g+1);
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
Q=sparse(iii,jjj,Q_s,L*J*K*(d1+1)*(d2+1)*(d2+2)/2,L*J*K*(d1+1)*(d2+1)*(d2+2)/2);
Q=(Q+Q')/2;

%%%Differentability Matrix: H%%%
H_1=zeros((rho+1)*(d1+1)*(2*d2+2-rho)/2,(d1+1)*(d2+1)*(d2+2)/2);
for h=0:d1
    for t=0:rho
        for m=0:d2-t
            for g=t:d2-m
                H_1(h*(rho+1)*(d2+(2-rho)/2)+t*(d2+(3-t)/2)+m+1,h*(d2+1)*(d2+2)/2+m*(d2+(3-m)/2)+g+1)=factorial(g)/((delta_x^t)*factorial(g-t));
            end
        end
    end
end

H_2=zeros((rho+1)*(d1+1)*(2*d2+2-rho)/2,(d1+1)*(d2+1)*(d2+2)/2);
for h=0:d1
    for t=0:rho
        for m=0:d2-t
            H_2(h*(rho+1)*(d2+(2-rho)/2)+t*(d2+(3-t)/2)+m+1,h*(d2+1)*(d2+2)/2+m*(d2+(3-m)/2)+t+1)=-factorial(t)/delta_x^t;
        end
    end
end

H_3=zeros((rho+1)*(d1+1)*(2*d2+2-rho)/2,(d1+1)*(d2+1)*(d2+2)/2);
for h=0:d1
    for t=0:rho
        for g=0:d2-t
            for m=t:d2-g
                H_3(h*(rho+1)*(d2+(2-rho)/2)+t*(d2+(3-t)/2)+g+1,h*(d2+1)*(d2+2)/2+m*(d2+(3-m)/2)+g+1)=factorial(m)/((delta_y^t)*factorial(m-t));
            end
        end
    end
end

H_4=zeros((rho+1)*(d1+1)*(2*d2+2-rho)/2,(d1+1)*(d2+1)*(d2+2)/2);
for h=0:d1
    for t=0:rho
        for g=0:d2-t
            H_4(h*(rho+1)*(d2+(2-rho)/2)+t*(d2+(3-t)/2)+g+1,h*(d2+1)*(d2+2)/2+t*(d2+(3-t)/2)+g+1)=-factorial(t)/delta_y^t;
        end
    end
end
iii=zeros(L*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4,1);
jjj=zeros(L*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4,1);
H_s=zeros(L*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4,1);

for kk=1:L% Number of spline
    for ii=1:2*(2*J*K-J-K)% Number of matirx
        if ii<=K*(J-1)% H1
            pos = fix((ii-1)/(J-1));
            for row=1:(rho+1)*(d1+1)*(2*d2+2-rho)/2% Number of row
                for col=1:(d1+1)*(d2+1)*(d2+2)/2% Number of column
                    iii((kk-1)*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (ii-1)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = (kk-1)*(2*J*K-J-K)*(rho+1)*(d1+1)*(2*d2+2-rho)/2 + (ii-1)*(rho+1)*(d1+1)*(2*d2+2-rho)/2 + row;
                    jjj((kk-1)*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (ii-1)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = (kk-1)*J*K*(d1+1)*(d2+1)*(d2+2)/2 + (ii-1+pos)*(d1+1)*(d2+1)*(d2+2)/2 + col;
                    H_s((kk-1)*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (ii-1)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = H_1(row,col);
                end
            end
        elseif ii<=2*K*(J-1)% H2
            pos = fix((ii-K*(J-1)-1)/(J-1));
            for row=1:(rho+1)*(d1+1)*(2*d2+2-rho)/2% Number of row
                for col=1:(d1+1)*(d2+1)*(d2+2)/2% Number of column
                    iii((kk-1)*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (ii-1)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = (kk-1)*(2*J*K-J-K)*(rho+1)*(d1+1)*(2*d2+2-rho)/2 + (ii-K*(J-1)-1)*(rho+1)*(d1+1)*(2*d2+2-rho)/2 + row;
                    jjj((kk-1)*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (ii-1)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = (kk-1)*J*K*(d1+1)*(d2+1)*(d2+2)/2 + (ii-K*(J-1)+pos)*(d1+1)*(d2+1)*(d2+2)/2 + col;
                    H_s((kk-1)*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (ii-1)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = H_2(row,col);
                end
            end
        elseif ii<=2*K*(J-1)+J*(K-1)% H3
            for row=1:(rho+1)*(d1+1)*(2*d2+2-rho)/2% Number of row
                for col=1:(d1+1)*(d2+1)*(d2+2)/2% Number of column
                    iii((kk-1)*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (ii-1)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = (kk-1)*(2*J*K-J-K)*(rho+1)*(d1+1)*(2*d2+2-rho)/2 + (ii-K*(J-1)-1)*(rho+1)*(d1+1)*(2*d2+2-rho)/2 + row;
                    jjj((kk-1)*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (ii-1)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = (kk-1)*J*K*(d1+1)*(d2+1)*(d2+2)/2 + (ii-2*K*(J-1)-1)*(d1+1)*(d2+1)*(d2+2)/2 + col;
                    H_s((kk-1)*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (ii-1)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = H_3(row,col);
                end
            end
        else% H4
            for row=1:(rho+1)*(d1+1)*(2*d2+2-rho)/2% Number of row
                for col=1:(d1+1)*(d2+1)*(d2+2)/2% Number of column
                    iii((kk-1)*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (ii-1)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = (kk-1)*(2*J*K-J-K)*(rho+1)*(d1+1)*(2*d2+2-rho)/2 + (ii-K*(J-1)-J*(K-1)-1)*(rho+1)*(d1+1)*(2*d2+2-rho)/2 + row;
                    jjj((kk-1)*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (ii-1)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = (kk-1)*J*K*(d1+1)*(d2+1)*(d2+2)/2 + (ii-2*K*(J-1)-J*(K-1)+J-1)*(d1+1)*(d2+1)*(d2+2)/2 + col;
                    H_s((kk-1)*2*(2*J*K-J-K)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (ii-1)*(rho+1)*(2*d2+2-rho)*(d1+1)^2*(d2+1)*(d2+2)/4 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = H_4(row,col);
                end
            end
        end
    end
end

H=sparse(iii,jjj,H_s,L*(2*J*K-J-K)*(rho+1)*(d1+1)*(2*d2+2-rho)/2,L*J*K*(d1+1)*(d2+1)*(d2+2)/2);


%%%Noncrossing Constraint Matrix: G%%%
G_1=zeros((d1+1)*(d2+1)*(d2+1),(d1+1)*(d2+1)*(d2+2)/2);
for l=0:d1
    for r=0:d2
        for v=0:d2
            for h=0:d1
                for m=0:d2
                    for g=0:d2-m
                        if h<=l && m<=v && g<=min(r,d2-m)
                            G_1(l*(d2+1)*(d2+1)+v*(d2+1)+r+1,h*(d2+1)*(d2+2)/2+m*(d2+(3-m)/2)+g+1) = factorial(d1-h)*factorial(d2-m)*factorial(d2-g)/(factorial(d1-l)*factorial(l-h)*factorial(d2-v)*factorial(v-m)*factorial(d2-r)*factorial(r-g));
                        end
                    end
                end
            end
        end
    end
end

iii=zeros(J*K*(d1+1)^2*(d2+1)^3*(d2+2),1);
jjj=zeros(J*K*(d1+1)^2*(d2+1)^3*(d2+2),1);
G_s=zeros(J*K*(d1+1)^2*(d2+1)^3*(d2+2),1);
for ii=1:J*K
    for row=1:(d1+1)*(d2+1)*(d2+1)
        for col=1:(d1+1)*(d2+1)*(d2+2)/2
            iii((ii-1)*(d1+1)^2*(d2+1)^3*(d2+2)/2 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = (ii-1)*(d1+1)*(d2+1)*(d2+1) + row;
            jjj((ii-1)*(d1+1)^2*(d2+1)^3*(d2+2)/2 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = (ii-1)*(d1+1)*(d2+1)*(d2+2)/2 + col;
            G_s((ii-1)*(d1+1)^2*(d2+1)^3*(d2+2)/2 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = G_1(row,col);

            iii(J*K*(d1+1)^2*(d2+1)^3*(d2+2)/2 + (ii-1)*(d1+1)^2*(d2+1)^3*(d2+2)/2 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = (ii-1)*(d1+1)*(d2+1)*(d2+1) + row;
            jjj(J*K*(d1+1)^2*(d2+1)^3*(d2+2)/2 + (ii-1)*(d1+1)^2*(d2+1)^3*(d2+2)/2 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = J*K*(d1+1)*(d2+1)*(d2+2)/2 + (ii-1)*(d1+1)*(d2+1)*(d2+2)/2 + col;
            G_s(J*K*(d1+1)^2*(d2+1)^3*(d2+2)/2 + (ii-1)*(d1+1)^2*(d2+1)^3*(d2+2)/2 + (row-1)*(d1+1)*(d2+1)*(d2+2)/2 + col) = -G_1(row,col);   
        end
    end
end

G=sparse(iii,jjj,G_s,J*K*(d1+1)*(d2+1)^2,J*K*(d1+1)*(d2+1)*(d2+2));


%% Solved by ADMM
warning('off', 'MATLAB:nearlySingularMatrix');
gamma = 1;
[LLL1,UUU1] = lu(2*lambda*Q + (A'*A+ G'*G)/gamma);
[LLL2,UUU2] = lu( H*(UUU1\(LLL1\(H'))));

c_k = rand(L*J*K*(d1+1)*(d2+1)*(d2+2)/2, 1);
z1_k = A*c_k;
z2_k = G*c_k;

u1_k = z1_k;
u2_k = z2_k;
dif1 = 1;
dif2 = 1;
dif3 = 1;
iteration = 1;
while dif1 > 10^(-5) || dif2 > 10^(-5) || dif3 > 10^(-5)
    iteration = iteration + 1;
    if mod(iteration, 100) == 0
        fprintf('ADMM iteration: %d / %d\n', iteration, iter);
    end
    aaa = (A'*(z1_k-u1_k)+G'*(z2_k-u2_k))/gamma;
    c_k1 = UUU1\(LLL1\(aaa - H'*(UUU2\(LLL2\(H*(UUU1\(LLL1\aaa)))))));
    z1_k1 = A*c_k1 + u1_k;

    %%%Upper Surface%%%
    for I = 1:n_s
        z1_k1(I) = per_prox(z1_k1(I) ,sort_data(I,4) ,p_upper(I),gamma*w(I));
    end
    %%%Lower Surface%%%
    for U = 1:n_s
        z1_k1(n_s + U) = per_prox(z1_k1(n_s + U) ,sort_data(U,4) ,p_lower(U),gamma*w(U));
    end

    z2_k1 = G*c_k1 + u2_k;
    z2_k1(z2_k1<0)=0;

    u1_k1 = u1_k + A*c_k1 -  z1_k1;
    u2_k1 = u2_k + G*c_k1 -  z2_k1;

    eps0 = 1e-12;
    dif1 = norm(c_k1 - c_k) / max([norm(c_k1), norm(c_k), eps0]);
    dif2 = norm([z1_k1; z2_k1] - [z1_k; z2_k]) / max([norm([z1_k1; z2_k1]), norm([z1_k; z2_k]), eps0]);
    dif3 = norm([u1_k1; u2_k1] - [u1_k; u2_k]) / max([norm([u1_k1; u2_k1]), norm([u1_k; u2_k]), eps0]);
    c_k = c_k1;
    z1_k = z1_k1;
    z2_k = z2_k1;
    u1_k = u1_k1;
    u2_k = u2_k1;
    if iteration>iter
        break
    end 
end
warning('on', 'MATLAB:nearlySingularMatrix');

%% Output
c_star = c_k;

n_coef_piece = (d1 + 1) * (d2 + 1) * (d2 + 2) / 2;

fineness = 100 / J;

tttt = linspace(0, 1, 11);

xxxx = spline_knots_X(1):delta_x/fineness:spline_knots_X(end);
yyyy = spline_knots_Y(1):delta_y/fineness:spline_knots_Y(end);
[xGrid, yGrid] = meshgrid(xxxx, yyyy);

%%%First pass: determine global color limits

width_min = inf;
width_max = -inf;
upper_min = inf;
upper_max = -inf;
lower_min = inf;
lower_max = -inf;

for tau_t = tttt

    zGrid = zeros(fineness*K+1, fineness*J+1, L);

    for iii = 1:L
        for jj = 1:J
            for kk = 1:K
                for j = 0:(fineness-1)
                    for k = 0:(fineness-1)

                        tau_x = (xxxx((jj-1)*fineness+j+1) - spline_knots_X(jj)) / delta_x;
                        tau_y = (yyyy((kk-1)*fineness+k+1) - spline_knots_Y(kk)) / delta_y;

                        for t_d = 0:d1
                            for k_d = 0:d2
                                for j_d = 0:(d2-k_d)

                                    idx = (iii-1)*(J*K*(d1+1)*(d2+1)*(d2+2)/2) ...
                                        + (kk-1)*(J*(d1+1)*(d2+1)*(d2+2)/2) ...
                                        + (jj-1)*((d1+1)*(d2+1)*(d2+2)/2) ...
                                        + t_d*(d2+1)*(d2+2)/2 ...
                                        + k_d*(d2+(3-k_d)/2) ...
                                        + j_d + 1;

                                    zGrid((kk-1)*fineness+k+1, ...
                                          (jj-1)*fineness+j+1, iii) = ...
                                    zGrid((kk-1)*fineness+k+1, ...
                                          (jj-1)*fineness+j+1, iii) ...
                                    + c_star(idx) * tau_x^j_d * tau_y^k_d * tau_t^t_d;

                                end
                            end
                        end
                    end
                end
            end
        end

        for jj = 1:J
            for j = 0:(fineness-1)

                tau_x = (xxxx((jj-1)*fineness+j+1) - spline_knots_X(jj)) / delta_x;

                for t_d = 0:d1
                    for k_d = 0:d2
                        for j_d = 0:(d2-k_d)

                            idx = (iii-1)*(J*K*(d1+1)*(d2+1)*(d2+2)/2) ...
                                + (K-1)*(J*(d1+1)*(d2+1)*(d2+2)/2) ...
                                + (jj-1)*((d1+1)*(d2+1)*(d2+2)/2) ...
                                + t_d*(d2+1)*(d2+2)/2 ...
                                + k_d*(d2+(3-k_d)/2) ...
                                + j_d + 1;

                            zGrid(fineness*K+1, ...
                                  (jj-1)*fineness+j+1, iii) = ...
                            zGrid(fineness*K+1, ...
                                  (jj-1)*fineness+j+1, iii) ...
                            + c_star(idx) * tau_x^j_d * tau_t^t_d;

                        end
                    end
                end
            end
        end

        for kk = 1:K
            for k = 0:(fineness-1)

                tau_y = (yyyy((kk-1)*fineness+k+1) - spline_knots_Y(kk)) / delta_y;

                for t_d = 0:d1
                    for k_d = 0:d2
                        for j_d = 0:(d2-k_d)

                            idx = (iii-1)*(J*K*(d1+1)*(d2+1)*(d2+2)/2) ...
                                + (kk-1)*(J*(d1+1)*(d2+1)*(d2+2)/2) ...
                                + (J-1)*((d1+1)*(d2+1)*(d2+2)/2) ...
                                + t_d*(d2+1)*(d2+2)/2 ...
                                + k_d*(d2+(3-k_d)/2) ...
                                + j_d + 1;

                            zGrid((kk-1)*fineness+k+1, ...
                                  fineness*J+1, iii) = ...
                            zGrid((kk-1)*fineness+k+1, ...
                                  fineness*J+1, iii) ...
                            + c_star(idx) * tau_y^k_d * tau_t^t_d;

                        end
                    end
                end
            end
        end

        for k_d = 0:d2
            for t_d = 0:d1
                for j_d = 0:(d2-k_d)

                    idx = (iii-1)*(J*K*(d1+1)*(d2+1)*(d2+2)/2) ...
                        + (K-1)*(J*(d1+1)*(d2+1)*(d2+2)/2) ...
                        + (J-1)*((d1+1)*(d2+1)*(d2+2)/2) ...
                        + t_d*(d2+1)*(d2+2)/2 ...
                        + k_d*(d2+(3-k_d)/2) ...
                        + j_d + 1;

                    zGrid(fineness*K+1, fineness*J+1, iii) = ...
                    zGrid(fineness*K+1, fineness*J+1, iii) ...
                    + c_star(idx) * tau_t^t_d;

                end
            end
        end
    end

    zGrid = zGrid * Z_normalize;

    widthGrid = zGrid(:,:,1) - zGrid(:,:,2);

    width_min = min(width_min, min(widthGrid, [], 'all'));
    width_max = max(width_max, max(widthGrid, [], 'all'));

    upper_min = min(upper_min, min(zGrid(:,:,1), [], 'all'));
    upper_max = max(upper_max, max(zGrid(:,:,1), [], 'all'));

    lower_min = min(lower_min, min(zGrid(:,:,2), [], 'all'));
    lower_max = max(lower_max, max(zGrid(:,:,2), [], 'all'));
end

width_clim = [width_min, width_max];
upper_clim = [upper_min, upper_max];
lower_clim = [lower_min, lower_max];

% Video writers
v1 = VideoWriter('STMIR_interval_width.mp4', 'MPEG-4');
v2 = VideoWriter('STMIR_upper_bound.mp4', 'MPEG-4');
v3 = VideoWriter('STMIR_lower_bound.mp4', 'MPEG-4');

v1.FrameRate = 10;
v2.FrameRate = 10;
v3.FrameRate = 10;

open(v1);
open(v2);
open(v3);

for tau_t = tttt

    zGrid = zeros(fineness * K + 1, fineness * J + 1, L);

    for iii = 1:L
        for jj = 1:J
            for kk = 1:K
                for j = 0:(fineness - 1)
                    for k = 0:(fineness - 1)

                        tau_x = (xxxx((jj - 1) * fineness + j + 1) - spline_knots_X(jj)) / delta_x;
                        tau_y = (yyyy((kk - 1) * fineness + k + 1) - spline_knots_Y(kk)) / delta_y;

                        for t_d = 0:d1
                            for k_d = 0:d2
                                for j_d = 0:(d2 - k_d)

                                    coef_idx_local = ...
                                        t_d * (d2 + 1) * (d2 + 2) / 2 + ...
                                        k_d * (d2 + (3 - k_d) / 2) + ...
                                        j_d + 1;

                                    idx = ...
                                        (iii - 1) * J * K * n_coef_piece + ...
                                        (kk - 1) * J * n_coef_piece + ...
                                        (jj - 1) * n_coef_piece + ...
                                        coef_idx_local;

                                    zGrid((kk - 1) * fineness + k + 1, ...
                                          (jj - 1) * fineness + j + 1, iii) = ...
                                    zGrid((kk - 1) * fineness + k + 1, ...
                                          (jj - 1) * fineness + j + 1, iii) + ...
                                    c_star(idx) * tau_x^j_d * tau_y^k_d * tau_t^t_d;

                                end
                            end
                        end
                    end
                end
            end
        end

        for jj = 1:J
            for j = 0:(fineness - 1)

                tau_x = (xxxx((jj - 1) * fineness + j + 1) - spline_knots_X(jj)) / delta_x;

                for t_d = 0:d1
                    for k_d = 0:d2
                        for j_d = 0:(d2 - k_d)

                            coef_idx_local = ...
                                t_d * (d2 + 1) * (d2 + 2) / 2 + ...
                                k_d * (d2 + (3 - k_d) / 2) + ...
                                j_d + 1;

                            idx = ...
                                (iii - 1) * J * K * n_coef_piece + ...
                                (K - 1) * J * n_coef_piece + ...
                                (jj - 1) * n_coef_piece + ...
                                coef_idx_local;

                            zGrid(fineness * K + 1, ...
                                  (jj - 1) * fineness + j + 1, iii) = ...
                            zGrid(fineness * K + 1, ...
                                  (jj - 1) * fineness + j + 1, iii) + ...
                            c_star(idx) * tau_x^j_d * tau_t^t_d;

                        end
                    end
                end
            end
        end

        for kk = 1:K
            for k = 0:(fineness - 1)

                tau_y = (yyyy((kk - 1) * fineness + k + 1) - spline_knots_Y(kk)) / delta_y;

                for t_d = 0:d1
                    for k_d = 0:d2
                        for j_d = 0:(d2 - k_d)

                            coef_idx_local = ...
                                t_d * (d2 + 1) * (d2 + 2) / 2 + ...
                                k_d * (d2 + (3 - k_d) / 2) + ...
                                j_d + 1;

                            idx = ...
                                (iii - 1) * J * K * n_coef_piece + ...
                                (kk - 1) * J * n_coef_piece + ...
                                (J - 1) * n_coef_piece + ...
                                coef_idx_local;

                            zGrid((kk - 1) * fineness + k + 1, ...
                                  fineness * J + 1, iii) = ...
                            zGrid((kk - 1) * fineness + k + 1, ...
                                  fineness * J + 1, iii) + ...
                            c_star(idx) * tau_y^k_d * tau_t^t_d;

                        end
                    end
                end
            end
        end

        for k_d = 0:d2
            for t_d = 0:d1
                for j_d = 0:(d2 - k_d)

                    coef_idx_local = ...
                        t_d * (d2 + 1) * (d2 + 2) / 2 + ...
                        k_d * (d2 + (3 - k_d) / 2) + ...
                        j_d + 1;

                    idx = ...
                        (iii - 1) * J * K * n_coef_piece + ...
                        (K - 1) * J * n_coef_piece + ...
                        (J - 1) * n_coef_piece + ...
                        coef_idx_local;

                    zGrid(fineness * K + 1, fineness * J + 1, iii) = ...
                    zGrid(fineness * K + 1, fineness * J + 1, iii) + ...
                    c_star(idx) * tau_t^t_d;

                end
            end
        end
    end

    zGrid = zGrid * Z_normalize;

    %%%Video 1: Interval Width
    fig1 = figure(1); clf;
    pcolor(xGrid, yGrid, zGrid(:,:,1) - zGrid(:,:,2));
    title(sprintf('Interval Width   (t = %.2f)', tau_t), 'FontSize', 24);
    shading interp;
    colormap(map_width);
    widthGrid = zGrid(:,:,1) - zGrid(:,:,2);
    clim(width_clim);
    colorbar;
    set(gca, 'FontSize', 16);
    xlabel('X', 'FontSize', 18);
    ylabel('Y', 'FontSize', 18);
    set(fig1, 'Name', sprintf('Width   (t = %.2f)', tau_t), 'NumberTitle', 'off');
    axis equal;
    axis tight;
    drawnow;
    writeVideo(v1, getframe(fig1));


    %%%Video 2: Upper Bound
    fig2 = figure(2); clf;
    pcolor(xGrid, yGrid, zGrid(:,:,1));
    title(sprintf('Upper Bound   (t = %.2f)', tau_t), 'FontSize', 24);
    shading interp;
    colormap(map_upper);
    clim(upper_clim);
    colorbar;
    set(gca, 'FontSize', 16);
    xlabel('X', 'FontSize', 18);
    ylabel('Y', 'FontSize', 18);
    set(fig2, 'Name', sprintf('Upper   (t = %.2f)', tau_t), 'NumberTitle', 'off');
    axis equal;
    axis tight;
    drawnow;
    writeVideo(v2, getframe(fig2));


    %%%Video 3: Lower Bound
    fig3 = figure(3); clf;
    pcolor(xGrid, yGrid, zGrid(:,:,2));
    title(sprintf('Lower Bound   (t = %.2f)', tau_t), 'FontSize', 24);
    shading interp;
    colormap(map_lower);
    clim(lower_clim);
    colorbar;
    set(gca, 'FontSize', 16);
    xlabel('X', 'FontSize', 18);
    ylabel('Y', 'FontSize', 18);
    set(fig3, 'Name', sprintf('Lower   (t = %.2f)', tau_t), 'NumberTitle', 'off');
    axis equal;
    axis tight;
    drawnow;
    writeVideo(v3, getframe(fig3));

end

close(v1);
close(v2);
close(v3);

save('stmir_coefficient.mat','c_star');


%% mCWC
nCovered = 0;
sumWidth = 0;

n = length(test_X);
lower = zeros(n, 1);
upper = zeros(n, 1);

n_coef_piece = (d1 + 1) * (d2 + 1) * (d2 + 2) / 2;
n_coef_function = J * K * n_coef_piece;

for i = 1:n

    %%%Find the subdomain in the X-direction
    ii_x = 0;
    for x0 = start_value_x:delta_x:(end_value_x - delta_x)
        ii_x = ii_x + 1;

        if (x0 <= test_X(i)) && (test_X(i) < x0 + delta_x)
            tau_x = (test_X(i) - x0) / delta_x;
            break
        end
    end

    %%%Find the subdomain in the Y-direction
    ii_y = 0;
    for y0 = start_value_y:delta_y:(end_value_y - delta_y)
        ii_y = ii_y + 1;

        if (y0 <= test_Y(i)) && (test_Y(i) < y0 + delta_y)
            tau_y = (test_Y(i) - y0) / delta_y;
            break
        end
    end

    %%%Time coordinate
    tau_t = test_T(i);

    %%%Evaluate upper and lower functions
    for t_d = 0:d1
        for k_d = 0:d2
            for j_d = 0:(d2 - k_d)

                coef_idx_local = ...
                    t_d * (d2 + 1) * (d2 + 2) / 2 + ...
                    k_d * (d2 + (3 - k_d) / 2) + ...
                    j_d + 1;

                coef_idx_upper = ...
                    (1 - 1) * n_coef_function + ...
                    (ii_y - 1) * J * n_coef_piece + ...
                    (ii_x - 1) * n_coef_piece + ...
                    coef_idx_local;

                coef_idx_lower = ...
                    (2 - 1) * n_coef_function + ...
                    (ii_y - 1) * J * n_coef_piece + ...
                    (ii_x - 1) * n_coef_piece + ...
                    coef_idx_local;

                upper(i) = upper(i) + ...
                    c_star(coef_idx_upper) * tau_x^j_d * tau_y^k_d * tau_t^t_d;

                lower(i) = lower(i) + ...
                    c_star(coef_idx_lower) * tau_x^j_d * tau_y^k_d * tau_t^t_d;

            end
        end
    end

    %%%Rescale to the original scale
    lower(i) = lower(i) * Z_normalize;
    upper(i) = upper(i) * Z_normalize;

    %%%Coverage and width
    if (test_Z(i) <= upper(i)) && (test_Z(i) >= lower(i))
        nCovered = nCovered + 1;
    end

    sumWidth = sumWidth + (upper(i) - lower(i));
end

PICP = nCovered / n;
MPIW = sumWidth / n;
NMPIW = MPIW / Z_normalize;

if PICP < alpha
    mCWC_value = NMPIW * exp(-eta * (PICP - alpha));
else
    mCWC_value = NMPIW;
end

fprintf('mCWC = %.6f\n', mCWC_value);