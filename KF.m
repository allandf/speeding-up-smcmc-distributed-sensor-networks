function[Kalman] = KF(P)
% Kalman filter over Tot_t time steps with stacked measurements.

d = P.d;Tot_t = P.Tot_t;
Q = P.Q;
Nmeas = P.Nmeas;
H = P.H; R = P.R;
A = P.A; zz = P.zz;

% Build model parameters for stacked observations
MP.InitialMean = zeros(d,1);
MP.Dimension = d;
MP.VarNoise = kron(eye(Nmeas), R);                 % block-diag(R,...,R)
MP.NoTimeSteps = Tot_t;
MP.ObservationTransition = repmat(H, Nmeas, 1);    % stack H per measurement
MP.StateCovariance = Q;
MP.TransitionMatrix = A*eye(d);
[Kalman]=KalmanFilter(MP,zz);

end

function [Kalman]=KalmanFilter(ModelParam,Observation)
% Standard Kalman recursion (predict/update) with stacked measurements.

ObservationCovariance=ModelParam.VarNoise;
Kalman.PredictMean=zeros(ModelParam.Dimension,ModelParam.NoTimeSteps);
Kalman.PredictCov=zeros(ModelParam.Dimension,ModelParam.Dimension,ModelParam.NoTimeSteps);
Kalman.UpdateMean=zeros(ModelParam.Dimension,ModelParam.NoTimeSteps);
Kalman.UpdateCov=zeros(ModelParam.Dimension,ModelParam.Dimension,ModelParam.NoTimeSteps);

for t=1:ModelParam.NoTimeSteps
    % --- PREDICTION ----
    if t==1
        Kalman.PredictMean(:,t)=ModelParam.InitialMean;
        Kalman.PredictCov(:,:,t)=ModelParam.StateCovariance;
    else
        Kalman.PredictMean(:,t)=ModelParam.TransitionMatrix*Kalman.UpdateMean(:,t-1);
        Kalman.PredictCov(:,:,t)=ModelParam.TransitionMatrix*Kalman.UpdateCov(:,:,t-1)*ModelParam.TransitionMatrix'+ModelParam.StateCovariance;
    end
    % --- UPDATE ----
    Innov=Observation(:,t)-ModelParam.ObservationTransition*Kalman.PredictMean(:,t);
    InnovCov=ModelParam.ObservationTransition*Kalman.PredictCov(:,:,t)*ModelParam.ObservationTransition'+ObservationCovariance;
    Gain=Kalman.PredictCov(:,:,t)*ModelParam.ObservationTransition'*inv(InnovCov);

    Kalman.UpdateMean(:,t)=Kalman.PredictMean(:,t)+Gain*Innov;
    Kalman.UpdateCov(:,:,t)=Kalman.PredictCov(:,:,t)-Gain*ModelParam.ObservationTransition*Kalman.PredictCov(:,:,t);
end
end
