
function [pairInfoMean, pairInfoMat, means, covs] = pair_joint_method(pairInfoMean, pairInfoMat, obs, V, neib_num, neib_id, neib_incidence_id, neib_wt, alpha)
%
% [pairInfoMean, pairInfoMat, means, covs] = ...
%        pair_joint_method(pairInfoMean, pairInfoMat, obs, V, neib_num, ...
%                              neib_id, neib_incidence_id, neib_wt, alpha)
%
% @input:
%   pairInfoMean = (n x n-1) cell
%   pairInfoMat = (n x n-1) cell
%
%   n = number of nodes
%   m = number of edges
%   d = node dimension
%
%   pairInfoMean{i,j} = (2d x 1) = pairwise information mean between nodes i and j
%   pairInfoMat{i,j} = (2d x 2d) = pairwise information matrix
%
%   obs = (d x m) = observations for each edge
%   V = (d x d) = measurement noise covariance
%
%   neib_num = (n x 1) = number of neibs for node i
%   neib_id = cell(n,1) = ids of the neibs of node i
%   neib_incidence_id = cell(n,1) = edge ids outgoing from node i
%   neib_wt = cell(n,1); = weights used to average the information from the
%                           neibs of node i
%
%   alpha = weight used to average the pairwise information
%
%
% @author:
%   Nikolay Atanasov
%   Grasp Lab
%   University of Pennsylvania
%   Sep 2013
%

n = size(pairInfoMean, 1);      % number of nodes
d = size(obs,1);                  % node dimension and number of edges

% Store the updated mean and covariance
new_pairInfoMean = cell(n,n-1);
new_pairInfoMat = cell(n,n-1);
means = zeros(n,d);
covs = cell(n);

% This is the distributed part
tmp = inv(V);   % kron([1 -1; -1 1], inv(V))
tmp = [tmp, -tmp; -tmp, tmp];
for i = 1:n


    % Incorporate neighbor information
    for j = 1:neib_num(i)
        jsid = neib_id{i}(j);
        js_idx_of_i = find(neib_id{jsid}==i);

        idx_reverse = [(d+1):(2*d), 1:d];

        new_pairInfoMat{i,j} = alpha*pairInfoMat{i,j} ...
            + (1-alpha)*pairInfoMat{jsid,js_idx_of_i}(idx_reverse, idx_reverse);

        new_pairInfoMean{i,j} = alpha*pairInfoMean{i,j} ...
            + (1-alpha)*pairInfoMean{jsid,js_idx_of_i}(idx_reverse);
    end


    % Average marginals
    MarginalInfoMean = cell(neib_num(i),1);
    MarginalInfoMat = cell(neib_num(i),1);

    SumInfoMat = zeros(d);
    SumInfoMean = zeros(d,1);
    for j = 1:neib_num(i)
        [MarginalInfoMean{j}, MarginalInfoMat{j}] = ...
            gaussInfoMarginal(new_pairInfoMean{i,j},new_pairInfoMat{i,j});

        SumInfoMat = SumInfoMat + neib_wt{i}(j)*MarginalInfoMat{j};
        SumInfoMean = SumInfoMean + neib_wt{i}(j)*MarginalInfoMean{j};
    end

    % Store the current estimates/marginals
    means(i,:) = SumInfoMat\SumInfoMean;
    covs{i} = inv(SumInfoMat);
        
    % Correct the joint
    for j = 1:neib_num(i)
        new_pairInfoMat{i,j}(1:d,1:d) = new_pairInfoMat{i,j}(1:d,1:d) ...
            - MarginalInfoMat{j} + SumInfoMat;

        new_pairInfoMean{i,j}(1:d) = new_pairInfoMean{i,j}(1:d) ...
            - MarginalInfoMean{j} + SumInfoMean;
    end


    % Include measurement info
    for j = 1:neib_num(i)        
        obs_tmp = V\obs(:,neib_incidence_id{i}(j)); %kron([-1;1],V\obs(:,neib_incidence_id{i}(j)));
        new_pairInfoMean{i,j} = new_pairInfoMean{i,j} + [-obs_tmp;obs_tmp];

        new_pairInfoMat{i,j} = new_pairInfoMat{i,j} + tmp;
    end

end

% update
pairInfoMat = new_pairInfoMat;
pairInfoMean = new_pairInfoMean;


end


function [vm, Im] = gaussInfoMarginal(v, I)

d = length(v)/2;
idxA = 1:d;
idxD = (d+1):2*d;

IA = I(idxA,idxA); 
IB = I(idxA,idxD);
ID = I(idxD,idxD);
tmp = IB/ID;


vm = v(idxA) - tmp*v(idxD);
Im = IA - tmp*IB';

end