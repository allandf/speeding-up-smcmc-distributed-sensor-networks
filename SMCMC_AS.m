function [xMCMC, accept, accept2, num_MU] = SMCMC_AS(P)
% SMCMC with Adaptive Subsampling MH step

N = P.N*P.S*P.K;H = P.H;R = P.R;z = P.Z;delta = 0.95;
x_actual = P.x_actual;Q = P.Q;A = P.A;
Tot_t = P.Tot_t; NBurn = P.NBurn; d = P.d;

initial = x_actual(1,:);
x{1} = cat(3,zeros(N+NBurn,d),mvnrnd(initial',Q*2,N+NBurn));
accept = zeros(1,Tot_t);accept2 = zeros(1,Tot_t);num_MU=zeros(Tot_t,N+NBurn);

Taylor.MatTmax = 1/R;

for t = 2:Tot_t
    fprintf('timestep = %i / %i\r',t,Tot_t);
    
    Taylor.point = (H*mean(A*(x{t-1}(:,:,2))',2))';
    Taylor.T1 = (z{t}-Taylor.point)/R;
    
    for n = 1:N+NBurn
        
        if n == 1
            x{t} = x{t-1}(NBurn+randi(N),:,2);        
            
            xP_init = mvnrnd((A*x{t}(n,:,1)')',Q);
            xP = mvnrnd((A*x{t}(n,:,1)')',Q);
            
            % Subsampled MH: returns phi, Lamda, and proxy correction Prox
            [phi,Lamda,Prox,num_MU(t,n),gamma,delta] = MHSubLHD(z{t},xP_init,xP,H,R,delta,Taylor);
            
            if Lamda > (phi - Prox) 
                x{t} = cat(3,x{t},xP);
            else
                x{t} = cat(3,x{t},xP_init);
            end
        else  
           
            xP = x{t-1}(NBurn+randi(N),:,2);
            GW = mvnpdf(x{t}(n-1,:,2),(A*xP')',Q)/...
                mvnpdf(x{t}(n-1,:,2),(A*x{t}(n-1,:,1)')',Q);
            if GW > rand
                x{t} = [x{t};cat(3,xP,zeros(1,d))];
                if n>NBurn
                    accept(t) = accept(t) + 1;
                end
            else
                x{t} = [x{t};cat(3,x{t}(n-1,:,1),zeros(1,d))];
            end
            
            xP = mvnrnd((A*x{t}(n,:,1)')',Q); 
            [phi,Lamda,Prox,num_MU(t,n)] = MHSubLHD(z{t},x{t}(n-1,:,2),xP,H,R,delta,Taylor);
            if Lamda > (phi - Prox)
                x{t}(n,:,2) = xP;
                accept2(t) = accept2(t) + 1;
            else
                x{t}(n,:,2) = x{t}(n-1,:,2);
            end
        end 
    end
end

for t = 1:Tot_t
    xMCMC(t,:) =  x{t}(NBurn+1:end,:,2);
    xMCMC_v(:,:,t) = cov(x{t}(NBurn+1:end,:,2)); 
end
end
