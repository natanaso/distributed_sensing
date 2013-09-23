% Functions needed:
% adjacency2incidence()
% gplot()

% True node locations:
xy = 25*rand(50,2);
xy_vec = xy';
xy_vec = xy_vec(:);

n = size(xy,1); % number of nodes
d = size(xy,2); % node dimension


% Connectivity
crad = 10;                      %connectivity radius
dMat = squareform(pdist(xy));
A = dMat < crad;                % Adjacency matrix
A = setdiag(A,zeros(n,1));      % no self loops
D = diag(sum(A,2));             % Degree matrix
L = D - A;                      % Laplacian
B = adjacency2incidence(A);     % Incidence matrix
B = [B, -B];
m = size(B,2);                  % number of edges

% Observation Model
z_m = zeros(1,m*d);             % noise mean
V = 0.1*eye(d) + 0.1*ones(d);   % noise covariance
V_full = kron(eye(m,m), V);     % full noise covariance (md x md)
G = kron(B,eye(d));
GX = G'*xy_vec;
z = @() GX + transpose(z_m + randn(1,m*d)*chol(V_full)); %z = GX + nv;


% parameters
alpha = 1/2;
neib_num = sum(A,2);
neib_id = cell(n,1);
neib_wt = cell(n,1);
neib_incidence_id = cell(n,1);
for i=1:n
    neib_id{i} = find(A(i,:));
    neib_wt{i} = 1/neib_num(i)*ones(1,neib_num(i));
    neib_incidence_id{i} = find(B(i,:) == -1);
end

% prior
key_node = 2;       % one node knows its location well
big_esp = 100000;

m_0 = xy + 4*randn(size(xy));
m_0(key_node,:) = xy(key_node,:);

S_0 = 4*eye(d);
I_0 = inv(S_0);
I_0_key = big_esp*eye(d);

% each node maintains a joint distribution
PairInfoMatrix = cell(n,n-1);
PairInfoMean = cell(n,n-1);

for i = 1:n
    
    for j = 1:neib_num(i)
        jsid = neib_id{i}(j);
        
        if( i == key_node)
            PairInfoMatrix{i,j} = blkdiag(I_0_key, I_0);
        elseif( jsid == key_node)
            PairInfoMatrix{i,j} = blkdiag(I_0, I_0_key);
        else
            PairInfoMatrix{i,j} = blkdiag(I_0, I_0);
        end
        
        PairInfoMean{i,j} = PairInfoMatrix{i,j}*reshape(transpose(m_0([i jsid],:)),[],1);
    end
end


% run algorithm for T steps
% Plot
figure;
gplot(A,xy,':rs');              % true positions (RED)
hold on;
xlabel('x');
ylabel('y');
title('Network topology');
[Xout,Yout] = gplot(A,m_0);
h = plot(Xout,Yout,':b*');      % Distributed estimates (BLUE)

profile on;
T = 20;
obs = cell(T,1);        % observations
for t = 1:T
    % new edge observations
    obs{t} = reshape(z(),d,m); 
    
    % make observations symmetric;
    obs{t}(:,(m/2+1):end) = -obs{t}(:,1:m/2);
    
    % received by node i:
    % obs{t}(:,neib_incidence_id{i})
    % z_ij = obs{t}(:,neib_incidence_id{i}(j));
    
    % Update the information with the new observation
    [PairInfoMean, PairInfoMatrix, means, covs] = pair_joint_method(PairInfoMean, PairInfoMatrix, ...
        obs{t}, V, neib_num, neib_id, neib_incidence_id, neib_wt, alpha);
    
    % plot
    [Xout,Yout] = gplot(A,means);
    set(h,'Xdata',Xout,'Ydata',Yout);  
end
profile viewer;


% Centralized Solution
M_cntr = G/V_full;
I_estm = @(I) I + M_cntr*G';
v_estm = @(v,z) M_cntr*z + v;   % Omega * mean

I_cntr_0 = kron(I_0,eye(n));
key_start = (key_node-1)*d+1;
key_end = key_node*d;
I_cntr_0(key_start:key_end,key_start:key_end) = I_0_key;

v_cntr_0 = I_cntr_0*reshape(transpose(m_0),[],1);

I_cntr = I_cntr_0;
v_cntr = v_cntr_0;
for t=1:T
    I_cntr = I_estm(I_cntr);
    v_cntr = v_estm(v_cntr,obs{t}(:));
end

m_cntr = I_cntr\v_cntr;
m_cntr = transpose(reshape(m_cntr,2,[]));
S_cntr = inv(I_cntr);

% plot
[Xout,Yout] = gplot(A,m_cntr);
h = plot(Xout,Yout,':g*');      % Centralized estimates (GREEN)
