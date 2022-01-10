clear; clc; close all
%% Sina Sep 2021
%%% ICASSP 2022 
%%% AdaJSR

%% Parameters
 seed=8;
 rng(seed);  
M= 100;    %% 
grid_size =100;  %% Dimention of X (d in paper)
K = 10;    %%%% Suport size

runs=30;  
maxx=12000;
nn=floor(logspace(log10(45),log10(maxx),runs));


de=ceil(log2(grid_size));
MC_run=10;
%L =  K*de;
L_max=75;
AR_g=1;
disp("Klog d:")
K*de
%SNR=0;
p =1;    %%% Signal Power
bias=1;   %%% Signal Mean (for non-zero elements)
  thr = 1e-8;  %%% Neccesary due to Matlab Computanial accuracy
%% Variables 

theta_grid=(0:grid_size-1)/grid_size;
%%%%%%  Random basis A
A= randn (M,grid_size);
%%%%% Antenna array A
%ula=0:M-1;
% A=exp(-(1i*2*pi).*ula'*theta_grid);
 %%%%%%%%%%  
jj=1;
for jj=1:runs     %%%% Number total accesible measurements 
n_max= nn(jj);
    err=zeros(MC_run,1);
    jj
    parfor mc=1:MC_run      %%%% Monte Carlo run
        fixed_w = randn(1,M);
        L = n_max+de;
        [Y,GT]=generate_y(A,L,M,grid_size,K,p,bias,AR_g);
        theta_grid(GT);

        We=cell(K,1);
        est=zeros(K,1);

        use=0;
        for k=1:K
            %%%% Calling AdaJSR recovery algorithm 

            [temp_est,temp,use]=find_doa(A,Y,use,est,thr,fixed_w);
            if use<=n_max    %%% Checking the total number of used measurements
                est(k)=temp_est;
            else
                break
            end
            %We{k}=temp;

        end   
        %%%%%%  Check if the correct support has been recovered
        if k==K
            err(mc)=norm(sort(GT)-sort(est'));
        else
            err(mc) = inf;
        end
    end
    %%% Compute the probability of error
     suc_prob(jj)=length(find(err==0))/MC_run;
     jj=jj+1;
end




%% Results and Plots


figure(1)
semilogx(nn,suc_prob,'LineWidth',4,'Marker','o','MarkerSize',5)
grid on
% xlim([4,40])
ylim([0,1.2])
legend('d=100, K=10, m=1')
ylabel('Probability of success')
xlabel('n')
title('AdaJSR')
set(gca,'Fontsize',20,'linewidth',3)

ax1=gca;
% ax1.GridColor = 'k'
ax1.GridAlpha = 0.35;


fname=strcat('AdaJSR_noniid_',num2str(K),'_sparsity_',num2str(M),'d_',num2str(maxx),'_end','.mat');
save(fname,'suc_prob','nn','M','K','-v7.3')

%% Functions
function my_plot(We,theta_grid,K)
    for k=1:K
        figure(k);
        temp_w=We{k,1};
        [sw,~]=size(temp_w);
        for i=1:sw
           plot([-1,theta_grid(1:end-1)],abs(fftshift(fft(temp_w(i,:)))),'LineWidth',3,'Marker','o','MarkerSize',2,'DisplayName',['Layer',num2str(i)])
          % plot(abs((temp_w(i,:))),'LineWidth',3,'Marker','o','MarkerSize',2,'DisplayName',['Layer',num2str(i)])
           hold on
            grid on
        end
         title(['Source',num2str(k)])
         xlabel('W')
         ylabel('Absolute Value')
         set(gca,'Fontsize',20,'linewidth',3)
         ax1=gca;
            % ax1.GridColor = 'k'
         ax1.GridAlpha = 0.35;
    end

end

function my_plot2(theta_grid,GT,est,M)
    figure(100)
    GT_plot=zeros(1,M);
    GT_plot(GT)=0.2;
    est_plot=zeros(1,M);
    est_plot(est)=0.1;
    stem(theta_grid,GT_plot,'r-.','LineWidth',3,'Marker','o','MarkerSize',5)
%     stem(GT,0.2*ones(length(GT)),'r-.','LineWidth',3,'Marker','o','MarkerSize',5,'DisplayName','Ground of Thruth')
    hold on
    stem(theta_grid,est_plot,'b-','LineWidth',3,'Marker','s','MarkerSize',5,'DisplayName','Estimated')
    grid on
     legend('Ground of Thruth','Estimated')
    title('DOA ESTIMATION')
    
     xlabel('Source Location')
     ylim([1 0.3]);
     set(gca,'Fontsize',20,'linewidth',3)
     ax1=gca;
        % ax1.GridColor = 'k'
     ax1.GridAlpha = 0.35;
end


function [Y,DOA]=generate_y(A,L,M,grid_size,K,p,bias,AR_g)
    DOA=sort(randperm(M,K));
   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian
%
 X_mat = abs(bias+sqrt(p)*randn(K,L)).*sign(randn(K,L));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% correlated
%  mu=zeros(K,1);
% U=randn(K);
% Sigma=U*U';
% X_mat = mvnrnd(mu,Sigma,L)'; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% non iid
%  X_vec= randn(K,1); 
%  X_coeff = randn(1,L);
%  X_mat= X_vec * X_coeff;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Autoregresive
%  X_vec= randn(K,1);
%   X_mat =zeros(K,L);
%   X_mat(:,1)= X_vec;
%   alpha=randn;
% for ll=2:L
%     X_mat(:,ll)= alpha* X_mat(:,ll-1)+randn(K,1);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Exponential distribution 

  %  X_mat= exprnd(1,K,L);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% measurment 
    X=zeros(grid_size,L);
    X(DOA,1:L)=X_mat;
    
    Y=A*X ;
end


function w=querry_biulder(A,G,ind_w,fixed_w)
    b_w = zeros(1,G);

   b_w(ind_w)=fixed_w(ind_w);
    w = b_w*A^(-1);

end

function [S_ind,W,use]=find_doa(A,Y_p,use,est,thr,fixed_w)
    [M,G]=size(A);
    S_ind = 1:G;
    [~,~,val] = find(est);
    S_ind = setdiff(S_ind,val);
    i=1;
    while length(S_ind)>1
        %%%%% # Measurement design
        sd=ceil(length(S_ind)/2);
        S_ind_l = S_ind(1:sd);
        S_ind_r = S_ind(sd+1:end);
        W_le=querry_biulder(A,G,S_ind_l,fixed_w);
        W_ri=querry_biulder(A,G,S_ind_r,fixed_w);
        
        
        tmp=abs(W_le*Y_p(:,i+use)); %%% The acquired measurement
        %%%% Decision making   
        if tmp > thr
            S_ind = S_ind_l;
            W(i,:) = W_le;
        else
            S_ind = S_ind_r;
            W(i,:) = W_ri;
        end
        i=i+1;
    end
    use=use+i-1;
    
end


