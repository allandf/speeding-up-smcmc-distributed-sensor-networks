function [meanv, accept2, accept] = SMCMC_EP(P)
% EP-based divide-and-conquer SMCMC 

N = P.N;S = P.S;K = P.K;
H = P.H;R = P.R;z = P.z;
x_actual = P.x_actual;Q = P.Q;A = P.A;
Tot_t = P.Tot_t; NBurn = P.NBurn; d = P.d;
for s = 1:S
N_min(s) = length(P.z{end,s}); 
end

% particle initialisation per shard
initial = x_actual(1,:);
x = cell(Tot_t,S);
for s=1:S
    x{1,s} = cat(3,zeros(N+NBurn,d),mvnrnd(initial,Q*2,N+NBurn));
end
x_total = cell(Tot_t,1);
for s = 1:S
    x_total{1} =[x_total{1};x{1,s}(NBurn+1:end,:,2)];
end
    meanv(1,:) = x_total{1}; 
    varv(:,:,1) = cov(x_total{1}); 


% variable initialisations 
accept = zeros(Tot_t,K,S);
accept2 = zeros(Tot_t,K,S);
pred_post = zeros(N,d,S);           
pred_post_mu = zeros(1,d,S);
pred_post_sigma = zeros(d,d,S);
pred_post_NP_p1 = zeros(1,d,S);     
pred_post_NP_p2 = zeros(d,d,S);     
prior_NP_p1 = zeros(1,d,S);
prior_NP_p2 = zeros(d,d,S);
tilted_mu = zeros(1,d,S);
tilted_sigma = zeros(d,d,S);
tilted_NP_p1 = zeros(1,d,S);
tilted_NP_p2 = zeros(d,d,S);
batch_NPnew_p1 = zeros(1,d,S);
batch_NPnew_p2 = zeros(d,d,S);

for t = 2:Tot_t
    for k = 1:K
        if k == 1
            % Initialize site natural params to zero
            batch_NP_p1 = zeros(1,d);
            batch_NP_p2 = zeros(d,d);
            batch_NP_p1 = repmat(batch_NP_p1,1,1,S);
            batch_NP_p2 = repmat(batch_NP_p2,1,1,S);
        end
        for s = 1:S
            x_prev{s} = x{t-1,s}; 
        end
        fprintf('timestep = %i / %i, EP loop = %i / %i.\r',t,Tot_t,k,K);
        parfor s = 1:S
            warning('error', 'MATLAB:illConditionedMatrix');
            warning('error', 'MATLAB:nearlySingularMatrix');
            warning('error', 'MATLAB:singularMatrix');
            if k == 1
                
                pred_post(:,:,s) = mvnrnd((A*x_prev{s}(NBurn+1:end,:,2)')',Q);
                pred_post_mu(:,:,s) = mean(pred_post(:,:,s));
                pred_post_sigma(:,:,s) = cov(pred_post(:,:,s));
                pred_post_NP_p1(:,:,s) = pred_post_mu(:,:,s)*pred_post_sigma(:,:,s)^-1;
                pred_post_NP_p2(:,:,s) = pred_post_sigma(:,:,s)^-1;
            end
            for n = 1:N+NBurn
                if n == 1
                    
                    x{t,s} = x_prev{s}(NBurn+randi(N),:,2);
                    if k == 1
                        xP_init = mvnrnd((A*x{t,s}(n,:,1)')',Q);
                        xP = mvnrnd((A*x{t,s}(n,:,1)')',Q);
                    else
                        mu = (A*x{t,s}(n,:,1)')';
                        prior_NP_p1(:,:,s) = mu*Q^-1; 
                        prior_NP_p2(:,:,s) = Q^-1;
                        
                        GNP_p1 = prior_NP_p1(:,:,s) + sum(batch_NP_p1(:,:,1:S~=s),3);
                        GNP_p2 = prior_NP_p2(:,:,s) + sum(batch_NP_p2(:,:,1:S~=s),3);
                        GNP_mu = GNP_p1*inv(GNP_p2);
                        GNP_sigma = GNP_p2^-1;
                        xP_init = mvnrnd(GNP_mu,GNP_sigma);
                        xP = mvnrnd(GNP_mu,GNP_sigma);
                    end
                    if k == 1
                    [phi,Lamda] = MHstepG_dist(z{t,s}(1:N_min(s)),xP_init,xP,H,R,k);
                    else
                    [phi,Lamda] = MHstepG_dist(z{t,s},xP_init,xP,H,R,k,GNP_mu,GNP_sigma,mu,Q,batch_NP_p1(:,:,1:S~=s),batch_NP_p2(:,:,1:S~=s));    
                    end
                    if Lamda > phi
                        x{t,s} = cat(3,x{t,s},xP);
                    else
                        x{t,s} = cat(3,x{t,s},xP_init);
                    end
                else
                    
                    if t == 1
                        x{t,s} = [x{t,s};cat(3,x_prev{s}(NBurn+randi(N),:,2),zeros(1,d))];
                    else
                        xP = x_prev{s}(NBurn+randi(N),:,2);
                          GW = mvnpdf(x{t,s}(n-1,:,2),(A*xP')',Q)/...
                               mvnpdf(x{t,s}(n-1,:,2),(A*x{t,s}(n-1,:,1)')',Q);
                        if GW > rand
                            x{t,s} = [x{t,s};cat(3,xP,zeros(1,d))];
                            if n>NBurn
                                accept2(t,k,s) = accept2(t,k,s) + 1;
                            end
                        else
                            x{t,s} = [x{t,s};cat(3,x{t,s}(n-1,:,1),zeros(1,d))];
                        end
                    end
                    
                    if k == 1
                        xP = mvnrnd((A*x{t,s}(n,:,1)')',Q);
                    else
                        mu = (A*x{t,s}(n,:,1)')';
                        prior_NP_p1(:,:,s) = mu*Q^-1; % natural parameters of the prior
                        prior_NP_p2(:,:,s) = Q^-1;
                        GNP_p1 = prior_NP_p1(:,:,s) + sum(batch_NP_p1(:,:,1:S~=s),3);
                        GNP_p2 = prior_NP_p2(:,:,s) + sum(batch_NP_p2(:,:,1:S~=s),3);
                        GNP_mu = GNP_p1*inv(GNP_p2);
                        GNP_sigma = GNP_p2^-1;
                        xP = mvnrnd(GNP_mu,GNP_sigma);
                    end
                    if k == 1
                    [phi,Lamda] = MHstepG_dist(z{t,s}(1:N_min(s)),x{t,s}(n-1,:,2),xP,H,R,k);
                    else 
                    [phi,Lamda] = MHstepG_dist(z{t,s},x{t,s}(n-1,:,2),xP,H,R,k,GNP_mu,GNP_sigma,mu,Q,batch_NP_p1(:,:,1:S~=s),batch_NP_p2(:,:,1:S~=s));
                    end
                    if Lamda > phi
                        x{t,s}(n,:,2) = xP;
                        if n>NBurn
                            accept(t,k,s) = accept(t,k,s) + 1;
                        end
                    else
                        x{t,s}(n,:,2) = x{t,s}(n-1,:,2);
                    end
                end
            end
            
            tilted_mu(:,:,s) = mean(x{t,s}(NBurn+1:end,:,2));
            tilted_sigma(:,:,s) = cov(x{t,s}(NBurn+1:end,:,2));
            tilted_NP_p1(:,:,s) = tilted_mu(:,:,s)*tilted_sigma(:,:,s)^-1;
            tilted_NP_p2(:,:,s) = tilted_sigma(:,:,s)^-1;
            batch_NPnew_p1(:,:,s) = tilted_NP_p1(:,:,s)-pred_post_NP_p1(:,:,s)-sum(batch_NP_p1(:,:,1:S~=s),3);
            batch_NPnew_p2(:,:,s) = tilted_NP_p2(:,:,s)-pred_post_NP_p2(:,:,s)-sum(batch_NP_p2(:,:,1:S~=s),3);
        end
        
        batch_NP_p1=batch_NPnew_p1;
        batch_NP_p2=batch_NPnew_p2;
        
        for s = 1:S
            x_total{t} =[x_total{t};x{t,s}(NBurn+1:end,:,2)];
        end
    end
end

for t = 2:Tot_t
    for k = 1:K
        mean_v{t}(k,:) = mean(x_total{t}(S*N*(k-1)+1:S*N*k,:));
        var_v{t}(:,:,k) = cov(x_total{t}(S*N*(k-1)+1:S*N*k,:));
        if k == K
            meanv(t,:) = x_total{t}(S*N*(k-1)+1:S*N*k,:);
            varv(:,:,t) = cov(x_total{t}(S*N*(k-1)+1:S*N*k,:)); 
        end
    end
end
fprintf('done!\r');
end
