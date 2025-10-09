function [phi,Lamda] = MHstepG_dist(z,x,xP,H,R,k,q_mu,q_sigma,prior_mu,prior_Q,batch_NP_p1,batch_NP_p2)
% Metropolis–Hastings log-acceptance ratio for EP-SMCMC.
% Inputs:
%   z               : measurement set 
%   x               : current state
%   xP              : proposed state
%   H, R            : linear measurement model and covariance
%   k               : EP iteration counter 
%   q_mu, q_sigma   : proposal mean/cov 
%   prior_mu,prior_Q: prior mean/cov 
%   batch_NP_p1,p2  : natural parameters for other sites/batches 
%
% Outputs:
%   phi   : log(U)/|z| threshold (U~Uniform)
%   Lamda : log acceptance ratio

phi = (1/size(z,1)) * (log(rand)); 

% Per-measurement Gaussian likelihoods under proposed/current
for jj = 1:length(z)
    likP(jj) = mvnpdf(z(jj),H.*xP,R);
    lik(jj) = mvnpdf(z(jj),H.*x,R);
end

if k == 1
Lamda = (1/size(z,1)) * sum(log(likP./lik));
else
batch_mu = reshape(batch_NP_p1.*(batch_NP_p2).^-1,length(batch_NP_p1),1);
batch_sigma = reshape(batch_NP_p2.^-1,length(batch_NP_p1),1);
for cc = 1:length(batch_mu)
    if cc == 1
    batchP =  mvnpdf(xP,batch_mu(cc),batch_sigma(cc)); 
    batch =  mvnpdf(x,batch_mu(cc),batch_sigma(cc));    
    else
    batchP = batchP * mvnpdf(xP,batch_mu(cc),batch_sigma(cc)); 
    batch = batch * mvnpdf(x,batch_mu(cc),batch_sigma(cc));
    end
end
% Full MH log-ratio:
Lamda = (1/size(z,1)) * sum(log((mvnpdf(x,q_mu,q_sigma).*mvnpdf(xP,prior_mu,prior_Q).*batchP.*likP)...
                                ./(mvnpdf(xP,q_mu,q_sigma).*mvnpdf(x,prior_mu,prior_Q).*batch.*lik)));  
end

end
