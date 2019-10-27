clc
clear

%time * 100
from = 5000; %starting point - 50 seconds
to = 25000; %ending point - 250 seconds
diff = to-from; %range

%loading first gas sample data
gA{1} = loadData('.\data1\B1_GEa_F100_R1.txt',from,to);
gA{2} = loadData('.\data1\B1_GEa_F100_R2.txt',from,to);
gA{3} = loadData('.\data1\B1_GEa_F100_R3.txt',from,to);
gA{4}= loadData('.\data1\B1_GEa_F100_R4.txt',from,to);
gA{5} = loadData('.\data1\B2_GEa_F100_R1.txt',from,to);
gA{6} = loadData('.\data1\B2_GEa_F100_R2.txt',from,to);
gA{7} = loadData('.\data1\B2_GEa_F100_R3.txt',from,to);
gA{8} = loadData('.\data1\B2_GEa_F100_R4.txt',from,to);
gA{9} = loadData('.\data1\B3_GEa_F100_R1.txt',from,to);
gA{10} = loadData('.\data1\B3_GEa_F100_R2.txt',from,to);
gA{11} = loadData('.\data1\B3_GEa_F100_R3.txt',from,to);
gA{12} = loadData('.\data1\B3_GEa_F100_R4.txt',from,to);
gA{13} = loadData('.\data1\B4_GEa_F100_R1.txt',from,to);
gA{14} = loadData('.\data1\B4_GEa_F100_R2.txt',from,to);

gA{15} = loadData('.\data1\B1_GEa_F090_R1.txt',from,to);
gA{16} = loadData('.\data1\B1_GEa_F090_R2.txt',from,to);
gA{17} = loadData('.\data1\B1_GEa_F090_R3.txt',from,to);
gA{18}= loadData('.\data1\B1_GEa_F090_R4.txt',from,to);
gA{19} = loadData('.\data1\B2_GEa_F090_R1.txt',from,to);
gA{20} = loadData('.\data1\B2_GEa_F090_R2.txt',from,to);
gA{21} = loadData('.\data1\B2_GEa_F090_R3.txt',from,to);
gA{22} = loadData('.\data1\B2_GEa_F090_R4.txt',from,to);
gA{23} = loadData('.\data1\B3_GEa_F090_R1.txt',from,to);
gA{24} = loadData('.\data1\B3_GEa_F090_R2.txt',from,to);
gA{25} = loadData('.\data1\B3_GEa_F090_R3.txt',from,to);
gA{26} = loadData('.\data1\B3_GEa_F090_R4.txt',from,to);
gA{27} = loadData('.\data1\B4_GEa_F090_R1.txt',from,to);
gA{28} = loadData('.\data1\B4_GEa_F090_R2.txt',from,to);

gA{29} = loadData('.\data1\B1_GEa_F080_R1.txt',from,to);
gA{30} = loadData('.\data1\B1_GEa_F080_R2.txt',from,to);
gA{31} = loadData('.\data1\B1_GEa_F080_R3.txt',from,to);
gA{32}= loadData('.\data1\B1_GEa_F080_R4.txt',from,to);
gA{33} = loadData('.\data1\B2_GEa_F080_R1.txt',from,to);
gA{34} = loadData('.\data1\B2_GEa_F080_R2.txt',from,to);
gA{35} = loadData('.\data1\B2_GEa_F080_R3.txt',from,to);
gA{36} = loadData('.\data1\B2_GEa_F080_R4.txt',from,to);
gA{37} = loadData('.\data1\B3_GEa_F080_R1.txt',from,to);
gA{38} = loadData('.\data1\B3_GEa_F080_R2.txt',from,to);
gA{39} = loadData('.\data1\B3_GEa_F080_R3.txt',from,to);
gA{40} = loadData('.\data1\B3_GEa_F080_R4.txt',from,to);
gA{41} = loadData('.\data1\B4_GEa_F080_R1.txt',from,to);
gA{42} = loadData('.\data1\B4_GEa_F080_R2.txt',from,to);

%load second gas sample data
gB{1} = loadData('.\data1\B1_GEy_F100_R1.txt',from,to);
gB{2} = loadData('.\data1\B1_GEy_F100_R2.txt',from,to);
gB{3} = loadData('.\data1\B1_GEy_F100_R3.txt',from,to);
gB{4} = loadData('.\data1\B1_GEy_F100_R4.txt',from,to);
gB{5} = loadData('.\data1\B2_GEy_F100_R1.txt',from,to);
gB{6} = loadData('.\data1\B2_GEy_F100_R2.txt',from,to);
gB{7} = loadData('.\data1\B2_GEy_F100_R3.txt',from,to);
gB{8} = loadData('.\data1\B2_GEy_F100_R4.txt',from,to);
gB{9} = loadData('.\data1\B3_GEy_F100_R1.txt',from,to);
gB{10} = loadData('.\data1\B3_GEy_F100_R2.txt',from,to);
gB{11} = loadData('.\data1\B3_GEy_F100_R3.txt',from,to);
gB{12} = loadData('.\data1\B3_GEy_F100_R4.txt',from,to);
gB{13} = loadData('.\data1\B4_GEy_F100_R1.txt',from,to);
gB{14} = loadData('.\data1\B4_GEy_F100_R2.txt',from,to);

gB{15} = loadData('.\data1\B1_GEy_F090_R1.txt',from,to);
gB{16} = loadData('.\data1\B1_GEy_F090_R2.txt',from,to);
gB{17} = loadData('.\data1\B1_GEy_F090_R3.txt',from,to);
gB{18}= loadData('.\data1\B1_GEy_F090_R4.txt',from,to);
gB{19} = loadData('.\data1\B2_GEy_F090_R1.txt',from,to);
gB{20} = loadData('.\data1\B2_GEy_F090_R2.txt',from,to);
gB{21} = loadData('.\data1\B2_GEy_F090_R3.txt',from,to);
gB{22} = loadData('.\data1\B2_GEy_F090_R4.txt',from,to);
gB{23} = loadData('.\data1\B3_GEy_F090_R1.txt',from,to);
gB{24} = loadData('.\data1\B3_GEy_F090_R2.txt',from,to);
gB{25} = loadData('.\data1\B3_GEy_F090_R3.txt',from,to);
gB{26} = loadData('.\data1\B3_GEy_F090_R4.txt',from,to);
gB{27} = loadData('.\data1\B4_GEy_F090_R1.txt',from,to);
gB{28} = loadData('.\data1\B4_GEy_F090_R2.txt',from,to);
gB{29} = loadData('.\data1\B1_GEy_F080_R1.txt',from,to);
gB{30} = loadData('.\data1\B1_GEy_F080_R2.txt',from,to);
gB{31} = loadData('.\data1\B1_GEy_F080_R3.txt',from,to);
gB{32}= loadData('.\data1\B1_GEy_F080_R4.txt',from,to);
gB{33} = loadData('.\data1\B2_GEy_F080_R1.txt',from,to);
gB{34} = loadData('.\data1\B2_GEy_F080_R2.txt',from,to);
gB{35} = loadData('.\data1\B2_GEy_F080_R3.txt',from,to);
gB{36} = loadData('.\data1\B2_GEy_F080_R4.txt',from,to);
gB{37} = loadData('.\data1\B3_GEy_F080_R1.txt',from,to);
gB{38} = loadData('.\data1\B3_GEy_F080_R2.txt',from,to);
gB{39} = loadData('.\data1\B3_GEy_F080_R3.txt',from,to);
gB{40} = loadData('.\data1\B3_GEy_F080_R4.txt',from,to);
gB{41} = loadData('.\data1\B4_GEy_F080_R1.txt',from,to);
gB{42} = loadData('.\data1\B4_GEy_F080_R2.txt',from,to);

%number of total samples
szgasA = size(gA(1,1:end));
szgasB = size(gB(1,1:end));

%normalize all points
ppm100Gas = vertcat(gA{1,1:end},gB{1,1:end});
norm100 = normalizeGasData(ppm100Gas);

%number of samples in each batch
numOfSamp = 16;
freq = diff/numOfSamp; %how often we take samples

%sample gas A
for(i=0:szgasA(2)-1)
s100A{i+1} = sampleMat(norm100((i)*diff+1:(i+1)*diff,:),freq,numOfSamp);
end

%sample gas B
for(i=szgasB(2):szgasA(2)+szgasB(2)-1)
s100B{i-(szgasA(2)-1)} = sampleMat(norm100((i)*diff+1:(i+1)*diff,:),freq,numOfSamp);
end

allS = vertcat(s100A{1,1:end},s100B{1,1:end});
szAll = size(allS);
szA = [szAll(1)/2 9];
szB = [szAll(1)/2 9];

%put actual y label
v = [1];
va = repelem(v,szA(1))';

w = [-1];
wa = repelem(w,szB(1))';

y = vertcat(va,wa);

%intialize data points
X(1:szAll(1),1:3)=0;

%sensors used
sA=4;
sB=9;

%set x values, 3rd column is constant -1 for bias term, so bias does not
%change in weight calculation during SGD algo
for (j=1:szAll(1))
    X(j,1) = allS(j,sA);
    X(j,2) = allS(j,sB);
    X(j,3) = -1;
end

figure;
gscatter(X(:,1),X(:,2),y,'rb','sx')
legend('Ea-H','Ey-H');
title('Two Gas Data Points');
xlabel ('sensor 3');
ylabel ('sensor 8');
%sensors 3,8

%regularization constant
lam = 0.001;

%number of iterations
[nsamples,nfeatures] = size(X);
fprintf('Number of samples : %f\n',nsamples);
it = nsamples*10;

%returns weights done by SGD
w_sgd = SVM_SGD(X,y,lam,it); % iteration more than 10 times the training set as per Andrew Ng recommendation 

%fprintf('%f %f %f\n', b.')

%calcuates scores at each test point
%{
for (j=1:nsamples)
   predictScore(X(j,1),X(j,2),b(1),b(2),b(3));
end
%}

%Do testing on whole grid with SGD weights
 d = 0.01; %frequency of x values
 [x1Grid,x2Grid] = meshgrid(0:d:1,0:d:1); %all data points
 xGrid = [x1Grid(:),x2Grid(:)];
 score = predictGrid(xGrid,w_sgd(1),w_sgd(2),w_sgd(3)); 
 %calculate score for all data points
 result = score';

%plot sgd SVM 

figure;
h(1:2)= gscatter(xGrid(:,1),xGrid(:,2),result,...
    [1 1 0; 0 1 1]);
hold on
h(3:4)= gscatter(X(:,1),X(:,2),y,'rb','sx');
title('{\bf Classification Regions - SGD }');
xlabel('sensor 3');
ylabel('sensor 8');
%legend('Predicted Ea-H','Predicted Ey-H','Ea-H','Ey-H');
legend(h,{'negative region','positive region', 'negative sample', 'positive sample'});
axis tight
[plm,nlm,mlm] = makeLines(-2,2,w_sgd(1),w_sgd(2),w_sgd(3));
hold on;
plm.LineWidth = 2;
nlm.LineWidth = 2;
mlm.LineWidth = 2;
axis([0 1 0 1]);

%SGD results
%fprintf('Num of iterations: %f\n',it);
%fprintf('SGD SVM results\n');
Ypredicted = scoresOfData(X,w_sgd(1),w_sgd(2),w_sgd(3));
[cor,incor] = accuracySVM(y,Ypredicted);
%fprintf('Regularization Constant: %f\n',lam);
%fprintf('Num of iterations: %f\n',it);
pcINC = (incor/(cor+incor))*100;
pcCOR = 100-pcINC;
%fprintf('Percent Correct : %f\n',pcCOR);
%fprintf('Percent Incorrect : %f\n',pcINC);

%--------------------START SMO
itmax = 10000; %num of iterations in SMO
C_val=4.75; %regularization constant
X(:,3) = []; %delete 3rd column, not needed in SMO
w_smo = SVM_SMO_Linear(X,y,C_val,0,'l',itmax); %return smo weights
w_smo = SVM_SMO_Linear(X,y,C_val,0,'l',itmax); %return smo weights
w_smo = SVM_SMO_Linear(X,y,C_val,0,'l',itmax); %return smo weights

%w_smo = SVM_SMO(X,y,C_val,0,'l',itmax); %return smo weights
%SMO results

%fprintf('SMO SVM results\n');
%fprintf('Num of its: %f\n',it);
%Ypredicted = scoresOfData(X,w_smo(1),w_smo(2),w_smo(3));
[cor,incor] = accuracySVM(y,Ypredicted);
%fprintf('Regularization Constant: %f\n',1/C_val);
%fprintf('Num of iterations: %f\n',it);
pcINC = (incor/(cor+incor))*100;
pcCOR = 100-pcINC;
%fprintf('Percent Correct : %f\n',pcCOR);
%fprintf('Percent Incorrect : %f\n',pcINC);

%------------------START API in Matlab

%y values
y = cell(szA(1)+szB(1),1);
y(1:szA(1),:) = {'A'};
y(szA(1)+1:szA(1)+szB(1),:) = {'B'};

%SVM API
fprintf('API SVM\n');
tic
SVMModel = fitcsvm(X(:,1:2),y);
toc 
tic
SVMModel = fitcsvm(X(:,1:2),y);
toc
tic
SVMModel = fitcsvm(X(:,1:2),y);
toc
%find support vectors
sv = SVMModel.SupportVectors;

%plot SVM API with support vectors

figure;
gscatter(X(:,1),X(:,2),y,'br','xs')
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
title('data points and support vectors  - API');
legend('Gas A','Gas B','Support Vector');
%legend('Ea-H','Ey-H','Support Vector');
xlabel('sensor 3');
ylabel('sensor 8');
hold on;

SVMModels = cell(2,1);
classes = unique(y);
rng(1); % For reproducibility

%svm hyperplane for API

for j = 1:numel(classes)
    indx = strcmp(y,classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(X(:,1:2),indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','linear','BoxConstraint',1);
end


%predict scores for test points 
d = 0.002;
[x1Grid,x2Grid] = meshgrid(0:d:1,0:d:1);
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);
Scores = zeros(N,numel(classes));

for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},xGrid);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);

%plot hyperplane generated by API svm
figure
h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
    [0 1 1; 1 1 0;]);
hold on
h(3:4) = gscatter(X(:,1),X(:,2),y,'br','xs');

legend(h,{'gas A region','gas B region',...
    'observed gas A','observed gas B'},...
    'Location','Northwest');
%{
legend(h,{'Ea-H region','Ey-H region',...
    'observed Ea-H','observed Ey-H'},...
    'Location','Northwest');
    %}
axis tight
title('API svm');
xlabel('sensor 3');
ylabel('sensor 8');
hold off

%gathering data for Neural network
%{
v = [1];
va = repelem(v,szA(1))';

w = [-1];
wa = repelem(w,szB(1))';

y = vertcat(va,wa);
%}
%{
X = X';
v = [1];
va = repelem(v,szA(1));

w = [0];
wa = repelem(w,szA(1));

u = vertcat(va,wa);

v = [0];
va = repelem(v,szA(1));

w = [1];
wa = repelem(w,szA(1));

t =  vertcat(va,wa);

y = horzcat(u,t);

dlmwrite('twogasXa.txt', X,' ')
dlmwrite('twogasYa.txt', y, ' ')
type twogasXa.txt
type twogasYa.txt
%}

clear;close all;
function [ws] = SVM_SGD(x,y,lambda,iterations)
[m,d] = size(x);
%fprintf('size of training data: %d\n', m);
ws = zeros(d,1);
%learning rate = 1/step
step = 100;
fprintf('SGD SVM\n');
tic
for i=1:iterations      % iterations over the full data set as per stochastic gradient descent algorithm
    x2 = [x y];
    %shuffle the data 
    random_x = x2(randperm(size(x2, 1)), :);
    x = random_x(:,1:end-1);
    y = random_x(:,end);
        for numsamples=1:m      % pick a single random data point
            if (y(numsamples)*dot(x(numsamples,:),ws) < 1) %if missclassified 
                ws = (1-1/step)*ws + 1/(lambda*step)*y(numsamples)*x(numsamples,:)'; %update weight with error
            else
                ws = (1-1/step)*ws; %update weigtht normally
            end

        step=step+1;         % increment counter, decrease learning rate
        end
end
    toc 
end 

function [w] = SVM_SMO(x,y,c,b,kt,its)

Osample=x;
Olabel=y;

N = length(y);
C = c; %Concluded after Cross-Validation
tol = 10e-5;
alpha = zeros(N,1); % initializing alphas as zero so that sum of alpha(i)*y(i) = 0 (one of the stationary conditions )
bias = b;
it=0;
maxit=its;
%  SMO Algorithm
% instead of waiting when aplhas can't be changed any more just let it be a fixed iteration
fprintf('SMO SVM\n');
tic
while (it<maxit)
    it = it +1;
    changed_alphas=0;
    N=size(y,1);
    for i=1:N % for each alpha(i) loop and pick another alpha chnage it and try to optimize the primal equation following all other constraint and not violating KKT condition
        Ei=sum(alpha.*y.*K(x,x(i,:),kt))-y(i); % calculate the prediction error Ei= f(x(i))-y(i) = w*x(i)-b -y(i)
        % substitue w from stationary equation w= sum(alpha*y*x)
        if ((Ei*y(i)<-tol) && alpha(i)<C)||(Ei*y(i) > tol && (alpha(i) > 0)) % kkt condition is violated 
            for j=[1:i-1,i+1:N] % pick an alpha which is not alpha(i)
                Ej=sum(alpha.*y.*K(x,x(j,:),kt))-y(j); % find the prediction error for j
                  alpha_iold=alpha(i); % preserve old alpha(i)
                  alpha_jold=alpha(j); % preserve old alpha(j)

                  if y(i)~=y(j)
                      
                      %if s=y1*y2 and s== -1 then alpha(i)-alpha(j) = gamma (const) 
                      %
                      
                      L=max(0,alpha(j)-alpha(i)); % max feasible values of alpha is constrained by C  
                      H=min(C,C+alpha(j)-alpha(i)); % min feasible values of alpha is constrained by C  
                  else 
                       %if s=y1*y2 and s== 1 then alpha(i)+ alpha(j) = gamma (const) 
                       % remember no alph acan be greater than C and can't be less than 0 as that
                       % was the original constraint of objective function
                       % so combining that equation maximum feasable value
                       % of alpha can be C or gamma -C depending on gamma <
                       % C or gamma > C
                      
                       % same for minimum feasable functions
                      L=max(0,alpha(i)+alpha(j)-C);
                      H=min(C,alpha(i)+alpha(j));
                  end

                  if (L==H)
                      continue
                  end
                  
                  eta = 2*K(x(j,:),x(i,:),kt)-K(x(i,:),x(i,:),kt)-K(x(j,:),x(j,:),kt); % eta = 2K12 - K11 -K22
                  % proof K12 = X1(transpose)*X2 K11= X1(transpose) * X1
                  % K22 = X2(transpose)* X2 then eta =
                  % -(X1-X2)(transpose)*(X1-X2) = -||X1-X2||**2 <=0
                  
                  if eta>=0 % neu is always less than zero 
                      continue
                  end
                  
                  %objective function in terms of alpha and prediction
                  %error is 1/2* neu*(alpha(j)**2 + ((y(j)*()-
                  %neu*alpha_jold) * alpha(j) + constant
                  
                  %so for optimization doing first derivative of  it w.r.t alpha(j) 
                  % gives the following equation whic gives how to update
                  % alpha(j) giving the relationship between old alpha(j)
                  % and new alpha(j) 
                  
                  alpha(j)=alpha(j)-( y(j)*(Ei-Ej) )/eta;
                  
                  % clipped value of alpha depending on maximum and minimum
                  % feasable limits set earlier by constraints 
                  
                  if alpha(j) > H
                      alpha(j) = H;
                  end
                  if alpha(j) < L
                      alpha(j) = L;
                  end

                  if norm(alpha(j)-alpha_jold,2) < tol % that is why toi value is very low as if the chnage in alpha is too low there is no point in updating.
                      continue
                  end
                  
                  alpha(i)=alpha(i)+y(i)*y(j)*(alpha_jold-alpha(j)); % as alpha(i) + s* alpha(j) = constant  so any change in alpha(j) 
                  %is to be compenstaed by adding or substracting s* chamne in alpha(j) to alpha(i) 
                  
                  
                  
                  b1 = bias - Ei - y(i)*(alpha(i)-alpha_iold)*K(x(i,:),x(i,:),kt)...
                      -y(j)*(alpha(j)-alpha_jold)*K(x(i,:),x(j,:),kt);
                  b2 = bias - Ej - y(i)*(alpha(i)-alpha_iold)*K(x(i,:),x(j,:),kt)...
                      -y(j)*(alpha(j)-alpha_jold)*K(x(j,:),x(j,:),kt);
           
            %update bias term
                  if 0<alpha(i)<C
                      bias=b1;
                  elseif 0<alpha(j)<C
                      bias=b2;
                  else
                      bias=(b1+b2)/2;
                  end
                  changed_alphas=changed_alphas+1;
            end
        end
    end
    if changed_alphas==0 %once alphas don't change, we reached most optimum point
        break
    end
    x=x((find(alpha~=0)),:);
    y=y((find(alpha~=0)),:);
    alpha=alpha((find(alpha~=0)),:);
end
toc
% Weights
W=sum(alpha.*y.*x);
% Bias
bias =mean( y - x*W');

w = [W bias];

Xsupport=x;Ysupport=y;
x=Osample;
y=Olabel;

%plot SMO SVM

figure;

scatter(x(y==1,1),x(y==1,2),'b')
hold on
scatter(x(y==-1,1),x(y==-1,2),'r')
hold on
scatter(Xsupport(Ysupport==1,1),Xsupport(Ysupport==1,2),'.b')
hold on
scatter(Xsupport(Ysupport==-1,1),Xsupport(Ysupport==-1,2),'.r')
hold on
[plm,nlm,mlm] = makeLines(0,1,W(1),W(2),-1*bias);
plm.LineWidth = 2;
nlm.LineWidth = 2;
mlm.LineWidth = 2;

axis([0 1 0 1])
title('SVM - SMO');
xlabel ('sensor 3');
ylabel ('sensor 8');


end 

function [w] = SVM_SMO_Linear(x,y,c,b,kt,its)

Osample=x;
Olabel=y;

N = length(y);
C = c; %Concluded after Cross-Validation
tol = 10e-5;
alpha = zeros(N,1); % initializing alphas as zero so that sum of alpha(i)*y(i) = 0 (one of the stationary conditions )
bias = b;
it=0;
maxit=its;
%  SMO Algorithm
% instead of waiting when aplhas can't be changed any more just let it be a fixed iteration
fprintf('SMO SVM\n');
tic
while (it<maxit)
    it = it +1;
    changed_alphas=0;
    N=size(y,1);
    for i=1:N % for each alpha(i) loop and pick another alpha chnage it and try to optimize the primal equation following all other constraint and not violating KKT condition
        xi = x(i,:);
        xiT = xi';
        yi = y(i);
        Ei=sum(alpha.*y.*x*xiT)-yi; % calculate the prediction error Ei= f(x(i))-y(i) = w*x(i)-b -y(i)
        % substitue w from stationary equation w= sum(alpha*y*x)
        if ((Ei*yi<-tol) && alpha(i)<C)||(Ei*yi > tol && (alpha(i) > 0)) % kkt condition is violated 
            for j=[1:i-1,i+1:N] % pick an alpha which is not alpha(i)
                xj = x(j,:);
                xjT = xj';
                yj = y(j);
                
                Ej=sum(alpha.*y.*x*xjT)-yj; % find the prediction error for j
                  alpha_iold=alpha(i); % preserve old alpha(i)
                  alpha_jold=alpha(j); % preserve old alpha(j)

                  if yi~=yj
                      
                      %if s=y1*y2 and s== -1 then alpha(i)-alpha(j) = gamma (const) 
                      
                      L=max(0,alpha(j)-alpha(i)); % max feasible values of alpha is constrained by C  
                      H=min(C,C+alpha(j)-alpha(i)); % min feasible values of alpha is constrained by C  
                  else 
                       %if s=y1*y2 and s== 1 then alpha(i)+ alpha(j) = gamma (const) 
                       % remember no alph acan be greater than C and can't be less than 0 as that
                       % was the original constraint of objective function
                       % so combining that equation maximum feasable value
                       % of alpha can be C or gamma -C depending on gamma <
                       % C or gamma > C
                      
                       % same for minimum feasable functions
                      L=max(0,alpha(i)+alpha(j)-C);
                      H=min(C,alpha(i)+alpha(j));
                  end

                  if (L==H)
                      continue
                  end
                  
                  etaA = xi;
                  etaAT = etaA';
                  etaB = xj;
                  etaBT = etaB';
                  K12 = etaB*etaAT;
                  K11 = etaA*etaAT;
                  K22 = etaB*etaBT;
                  
                  eta = 2*K12-K11-K22; % eta = 2K12 - K11 -K22
                  % proof K12 = X1(transpose)*X2 K11= X1(transpose) * X1
                  % K22 = X2(transpose)* X2 then eta =
                  % -(X1-X2)(transpose)*(X1-X2) = -||X1-X2||**2 <=0
                  
                  if eta>=0 % neu is always less than zero 
                      continue
                  end
                  
                  %objective function in terms of alpha and prediction
                  %error is 1/2* neu*(alpha(j)**2 + ((y(j)*()-
                  %neu*alpha_jold) * alpha(j) + constant
                  
                  %so for optimization doing first derivative of  it w.r.t alpha(j) 
                  % gives the following equation whic gives how to update
                  % alpha(j) giving the relationship between old alpha(j)
                  % and new alpha(j) 
                  
                  alpha(j)=alpha(j)-( yj*(Ei-Ej) )/eta;
                  
                  % clipped value of alpha depending on maximum and minimum
                  % feasable limits set earlier by constraints 
                  
                  if alpha(j) > H
                      alpha(j) = H;
                  end
                  if alpha(j) < L
                      alpha(j) = L;
                  end
                  
                  if norm(alpha(j)-alpha_jold,2) < tol % that is why toi value is very low as if the chnage in alpha is too low there is no point in updating.
                      continue
                  end
                  
                  alpha(i)=alpha(i)+yi*yj*(alpha_jold-alpha(j)); % as alpha(i) + s* alpha(j) = constant  so any change in alpha(j) 
                  %is to be compenstaed by adding or substracting s* chamne in alpha(j) to alpha(i) 
                  
                  
                  
                  b1 = bias - Ei - yi*(alpha(i)-alpha_iold)*K11...
                      -yj*(alpha(j)-alpha_jold)*K12;
                  b2 = bias - Ej - yi*(alpha(i)-alpha_iold)*K12...
                      -yj*(alpha(j)-alpha_jold)*K22;
           
            %update bias term
                  if 0<alpha(i)<C
                      bias=b1;
                  elseif 0<alpha(j)<C
                      bias=b2;
                  else
                      bias=(b1+b2)/2;
                  end
                  changed_alphas=changed_alphas+1;
            end
        end
    end
    if changed_alphas==0 %once alphas don't change, we reached most optimum point
        break
    end
    x=x((find(alpha~=0)),:);
    y=y((find(alpha~=0)),:);
    alpha=alpha((find(alpha~=0)),:);
end
toc
% Weights
W=sum(alpha.*y.*x);
% Bias
bias =mean( y - x*W');

w = [W bias];

Xsupport=x;Ysupport=y;
x=Osample;
y=Olabel;

%plot SMO SVM

figure;
d = 0.01; %frequency of x values
 [x1Grid,x2Grid] = meshgrid(0:d:1,0:d:1); %all data points
 xGrid = [x1Grid(:),x2Grid(:)];
 score = predictGrid(xGrid,W(1),W(2),-1*bias); 
 %calculate score for all data points
 result = score';
 
gscatter(xGrid(:,1),xGrid(:,2),result,[1 1 0; 0 1 1]);
hold on
scatter(x(y==1,1),x(y==1,2),'b')
hold on
scatter(x(y==-1,1),x(y==-1,2),'r')
hold on
scatter(Xsupport(Ysupport==1,1),Xsupport(Ysupport==1,2),'.b')
hold on
scatter(Xsupport(Ysupport==-1,1),Xsupport(Ysupport==-1,2),'.r')
hold on
[plm,nlm,mlm] = makeLines(0,1,W(1),W(2),-1*bias);
plm.LineWidth = 2;
nlm.LineWidth = 2;
mlm.LineWidth = 2;

axis([0 1 0 1])
title('SVM - SMO');
legend('Preditced Ea-H','Predicted Ey-H','Ey-H','Ea-H','Support Vector');
xlabel ('sensor 3');
ylabel ('sensor 8');

end 

function val = predictScore(x,y,w1,w2,w3) % predict score for one example
    val = w1*x+w2*y-w3;
    %fprintf('x: %f , y: %f , val = %f\n',x,y,val);
end 

function val= predictGrid(xG,w1,w2,w3) % predict score for a grid 
    for k = 1:length(xG)
        r = predictScore(xG(k,1),xG(k,2),w1,w2,w3);
        if (r <0)
            val(k)=0;
        else
            val(k)=1;
        end 
    end 
    
end 

function [J] = sampleMat(mat,inc,sampSize) %sample from a matrix
    szData = size(mat);
    S = zeros(9,sampSize);
    
    for(h=2:9)
        numOfSamp = 1;
        for(i=1:inc:szData(1))
            if(numOfSamp < (sampSize+1))
                S(h,numOfSamp) = mat(i,h);  
                numOfSamp = numOfSamp + 1;  
            end
        end
    end
    J = S';
end

function [normMat] = normalizeGasData(mat) %normalize data
    matx = mat(:,2:end);
    N = matx- min(matx);
    De = max(matx)- min(matx);
    normMatx = N./De;
    normMat = [mat(:,1) normMatx]; 
end

function [p,n,m] = makeLines(xl,xh,w1,w2,w3) %make margins and hyperplane line equation
    hold on;
lx = linspace(xl,xh);
yp = ((-(1+w3))/(w2))/((1+w3)/(w1))*(lx)+((1+w3)/(w2));
yn = ((-(-1+w3))/(w2))/((-1+w3)/(w1))*(lx)+((-1+w3)/(w2));
ym = (-w1/w2)*(lx)+(w3/w2);
p = plot(lx,yp);
n = plot(lx,yn);
m = plot(lx,ym);
end

function [partMat] = loadData(file,from,to) %load data into matrix
data = importdata(file,' ');
partMat = data(from:to,:);
end

function [predictedScore] = scoresOfData(data,w1,w2,w3) %predict score of each point
predictedScore = zeros(1,1);
szDat = size(data);
    for(i=1:szDat(1))
        valRes = predictScore(data(i,1),data(i,2),w1,w2,w3);
        if(valRes<0)
            predictedScore(i,1)=-1;
        else
            predictedScore(i,1)=1;
        end
    end
end

function [c i] = accuracySVM(y,yp) %how many incorrect and correct classifications of an svm
c=0;
i=0;
szY = size(y);
    for(g=1:szY(1))
        if(y(g,1)==yp(g,1))
            c = c +1;
        else
            i = i +1;
        end
    end
end

function ker=K(Xtrain,x_i,k) %kernel function

if k=='g'
    for i=1:size(Xtrain,1)
        ker(i,1)=exp(-norm(Xtrain(i,:)-x_i)); %gaussian Kernel
    end
elseif k=='l'
    for i=1:size(Xtrain,1)
        ker(i,1)=Xtrain(i,:)*x_i'; %linear Kernel
    end
elseif k=='p'
    for i=1:size(Xtrain,1)
        ker(i,1)=(Xtrain(i,:)*x_i').^3; %poly3 Kernel
    end
end

end