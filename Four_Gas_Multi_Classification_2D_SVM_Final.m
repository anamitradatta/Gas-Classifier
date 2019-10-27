clear
from = 6000;
to = 16000;
diff = to-from;

%load gas data
gA{1} = loadData('.\data1\B1_GEa_F100_R1.txt',from,to);
gA{2} = loadData('.\data1\B1_GEa_F100_R2.txt',from,to);
gA{3} = loadData('.\data1\B1_GEa_F100_R3.txt',from,to);
gA{4}= loadData('.\data1\B1_GEa_F100_R4.txt',from,to);
gA{5} = loadData('.\data1\B2_GEa_F100_R1.txt',from,to);
gA{6} = loadData('.\data1\B2_GEa_F100_R2.txt',from,to);
gA{7} = loadData('.\data1\B2_GEa_F100_R3.txt',from,to);
gA{8} = loadData('.\data1\B2_GEa_F100_R4.txt',from,to);

gB{1} = loadData('.\data1\B1_GCO_F100_R1.txt',from,to);
gB{2} = loadData('.\data1\B1_GCO_F100_R2.txt',from,to);
gB{3} = loadData('.\data1\B1_GCO_F100_R3.txt',from,to);
gB{4} = loadData('.\data1\B1_GCO_F100_R4.txt',from,to);
gB{5} = loadData('.\data1\B2_GCO_F100_R1.txt',from,to);
gB{6} = loadData('.\data1\B2_GCO_F100_R2.txt',from,to);
gB{7} = loadData('.\data1\B2_GCO_F100_R3.txt',from,to);
gB{8} = loadData('.\data1\B2_GCO_F100_R4.txt',from,to);


gC{1} = loadData('.\data1\B1_GMe_F010_R1.txt',from,to);
gC{2} = loadData('.\data1\B1_GMe_F010_R2.txt',from,to);
gC{3} = loadData('.\data1\B1_GMe_F010_R3.txt',from,to);
gC{4}= loadData('.\data1\B1_GMe_F010_R4.txt',from,to);
gC{5} = loadData('.\data1\B2_GMe_F010_R1.txt',from,to);
gC{6} = loadData('.\data1\B2_GMe_F010_R2.txt',from,to);
gC{7} = loadData('.\data1\B2_GMe_F010_R3.txt',from,to);
gC{8} = loadData('.\data1\B2_GMe_F010_R4.txt',from,to);

gD{1} = loadData('.\data1\B1_GMe_F100_R1.txt',from,to);
gD{2} = loadData('.\data1\B1_GMe_F100_R2.txt',from,to);
gD{3} = loadData('.\data1\B1_GMe_F100_R3.txt',from,to);
gD{4} = loadData('.\data1\B1_GMe_F100_R4.txt',from,to);
gD{5} = loadData('.\data1\B2_GMe_F100_R1.txt',from,to);
gD{6} = loadData('.\data1\B2_GMe_F100_R2.txt',from,to);
gD{7} = loadData('.\data1\B2_GMe_F100_R3.txt',from,to);
gD{8} = loadData('.\data1\B2_GMe_F100_R4.txt',from,to);

szgasA = size(gA(1,1:end));
szgasB = size(gB(1,1:end));
szgasC = size(gC(1,1:end));
szgasD = size(gD(1,1:end));

%normalize gas data
highPPMGas = vertcat(gA{1,1:end},gB{1,1:end},gC{1,1:end},gD{1,1:end});
normHighPPM = normalizeGasData(highPPMGas);

numOfSamp = 5;
freq = diff/numOfSamp;

%do sampling
for(i=0:szgasA(2)-1)
    sHighgA{i+1} = sampleMat(normHighPPM((i)*diff+1:(i+1)*diff,:),freq,numOfSamp);
end


for(i=szgasA(2):szgasA(2)+szgasB(2)-1)
    sHighgB{i-(szgasA(2)-1)} = sampleMat(normHighPPM((i)*diff+1:(i+1)*diff,:),freq,numOfSamp);
end

for(i=szgasA(2)+szgasB(2):szgasA(2)+szgasB(2)+szgasC(2)-1)
    sHighgC{i-(szgasA(2)+szgasB(2)-1)} = sampleMat(normHighPPM((i)*diff+1:(i+1)*diff,:),freq,numOfSamp);
end

for(i=szgasA(2)+szgasB(2)+szgasC(2):szgasA(2)+szgasB(2)+szgasC(2)+szgasD(2)-1)
    sHighgD{i-(szgasA(2)+szgasB(2)+szgasC(2)-1)} = sampleMat(normHighPPM((i)*diff+1:(i+1)*diff,:),freq,numOfSamp);
end


allS = vertcat(sHighgA{1,1:end},sHighgB{1,1:end},sHighgC{1,1:end},sHighgD{1,1:end});
szAll = size(allS);
szA = [szAll(1)/4 9];
szB = [szAll(1)/4 9];
szC = [szAll(1)/4 9];
szD = [szAll(1)/4 9];

%choose sensor values
X(1:szAll(1),1:3)=0;
sA=4;
sB=7;
for (j=1:szAll(1))
    X(j,1) = allS(j,sA);
    X(j,2) = allS(j,sB);
    X(j,3) = -1;
end

[nsamples,nfeatures] = size(X);

o = cell(szA(1)+szB(1)+szC(1)+szD(1),1);
o(1:szA(1),:) = {'Ea-H'};
o(szA(1)+1:szA(1)+szB(1),:) = {'CO-H'};
o(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),:) = {'Me-L'};
o(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),:) = {'Me-H'};

figure;
gscatter(X(:,1),X(:,2),o,'brgm','xos+');
xlabel('sensor 3');
ylabel('sensor 6');
hold off;

%do SGD for four gases
lam = 0.001;
it=10000;
[b1 y1] = make_SVM_SGD_FourC(X,1,-1,-1,-1,szAll(1)/4,lam,it);
[b2 y2] = make_SVM_SGD_FourC(X,-1,1,-1,-1,szAll(1)/4,lam,it);
[b3 y3] = make_SVM_SGD_FourC(X,-1,-1,1,-1,szAll(1)/4,lam,it);
[b4 y4] = make_SVM_SGD_FourC(X,-1,-1,-1,1,szAll(1)/4,lam,it);
%---------------------------------SVM multi class
%predict scores using SGD weights 
d = 0.01;
 [x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
 min(X(:,2)):d:max(X(:,2)));
 xGrid = [x1Grid(:),x2Grid(:)];
 
 N = size(xGrid,1);
 Scores = zeros(N,4);
 
 [label,score] = predictGrid(xGrid,b1(1),b1(2),b1(3));
 Scores(:,1) = score; 
 [label,score] = predictGrid(xGrid,b2(1),b2(2),b2(3));
 Scores(:,2) = score; 
 [label,score] = predictGrid(xGrid,b3(1),b3(2),b3(3));
 Scores(:,3) = score; 
 [label,score] = predictGrid(xGrid,b4(1),b4(2),b4(3));
 Scores(:,4) = score; 
 
 [~,type] = max(Scores,[],2);

 v = [1];
 va = repelem(v,szA(1))';

 w = [2];
 wa = repelem(w,szB(1))';

 u = [3];
 ua = repelem(u,szC(1))';

 t = [4];
 ta = repelem(t,szD(1))';
 
 Y = vertcat(va,wa,ua,ta);
 
 figure
%predict grid scores
h(1:4) = gscatter(xGrid(:,1),xGrid(:,2),type,[1 1 0; 0 1 1; 1 1 1; 0 1 0]);

hold on
%plot points
h(5:8) = gscatter(X(:,1),X(:,2),Y,[ 0 0 1; 0 0 0; 1 0 1; 1 0 0; 0.5 0.2 0]);

title('{\bf Gas Classification Regions -SGD}');

xlabel('Sensor 3'); % column 2 in data file

ylabel('Sensor 6'); %column 5 in data file

legend(h,{'Ea-H region','CO-h region','Me-L region', 'Me-H region','observed Ea-H','observed CO-H','observed Me-L','observed Me-H'});

axis tight

hold off
%----------------SMO SVM
C_val=5; %regularization constant
i_bias=0; %inital bias
itmax = 10000; %number of iterations
X(:,3) = [];
%do SMO for four gases
wS1 = SVM_SMO_Linear(X(:,1:2),y1,C_val,i_bias,'l',itmax);
wS2 = SVM_SMO_Linear(X(:,1:2),y2,C_val,i_bias,'l',itmax);
wS3 = SVM_SMO_Linear(X(:,1:2),y3,C_val,i_bias,'l',itmax);
wS4 = SVM_SMO_Linear(X(:,1:2),y4,C_val,i_bias,'l',itmax);

%predict scores using SMO weights
d = 0.01;
 [x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
 min(X(:,2)):d:max(X(:,2)));
 xGrid = [x1Grid(:),x2Grid(:)];
 
 N = size(xGrid,1);
 Scores = zeros(N,4);
 
 [label,score] = predictGrid(xGrid,wS1(1),wS1(2),wS1(3));
 Scores(:,1) = score; 
 [label,score] = predictGrid(xGrid,wS2(1),wS2(2),wS2(3));
 Scores(:,2) = score; 
 [label,score] = predictGrid(xGrid,wS3(1),wS3(2),wS3(3));
 Scores(:,3) = score; 
 [label,score] = predictGrid(xGrid,wS4(1),wS4(2),wS4(3));
 Scores(:,4) = score; 
 [~,type] = max(Scores,[],2);

 v = [1];
 va = repelem(v,szA(1))';

 w = [2];
 wa = repelem(w,szB(1))';

 u = [3];
 ua = repelem(u,szC(1))';

 t = [4];
 ta = repelem(t,szD(1))';
 
 Y = vertcat(va,wa,ua,ta);
 
 figure
 
 %predict scores of grid
h(1:4) = gscatter(xGrid(:,1),xGrid(:,2),type,[1 1 0; 0 1 1; 1 1 1; 0 1 0]);

hold on

%plot points
h(5:8) = gscatter(X(:,1),X(:,2),Y,[ 0 0 1; 0 0 0; 1 0 1; 1 0 0]);

title('{\bf Gas Classification Regions}');

xlabel('Sensor 3'); % column 2 in data file

ylabel('Sensor 6'); %column 5 in data file

legend(h,{'Ea-H region','CO-H region','Me-L region', 'Me-H region','observed Ea-H','observed CO-H','observed Me-L','observed Me-H'});

axis tight

hold off

%----------------API MATLAB------------------------

%choose sensor values
X(1:szAll(1),1:2)=0;
sA=4;
sB=7;
for (j=1:szAll(1))
    X(j,1) = allS(j,sA);
    X(j,2) = allS(j,sB);
end

[nsamples,nfeatures] = size(X);



y = cell(szA(1)+szB(1)+szC(1)+szD(1),1);
y(1:szA(1),:) = {'Ea-H'};
y(szA(1)+1:szA(1)+szB(1),:) = {'CO-H'};
y(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),:) = {'Me-L'};
y(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),:) = {'Me-H'};

SVMModels = cell(4,1);
classes = unique(y);
%rng(1); % For reproducibility

for j = 1:numel(classes)
    indx = strcmp(y,classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(X,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','linear','BoxConstraint',1);
end

d = 0.002;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);
Scores = zeros(N,numel(classes));

for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},xGrid);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);

figure
h(1:4) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
    [0 1 1; 1 1 0; 1 0 1;0 1 0]);
hold on
h(5:8) = gscatter(X(:,1),X(:,2),y,'brwg','xs+^');
legend(h,{'CO-H region','Ea-H region','Me-H region','Me-L region'...
    'observed Ea-H','observed CO-H','observed Me-L','observed Me-H'});
axis tight
hold off

function [w] = SVM_SGD(x,y,lambda,iterations)
[m,d] = size(x);
w = zeros(d,1);
step = 100;
for i=1:iterations      % iterations over the full data set as per stochastic gradient descent algorithm
    x2 = [x y];
    %shuffle the data 
    random_x = x2(randperm(size(x2, 1)), :);
    x = random_x(:,1:end-1);
    y = random_x(:,end);
    for numsamples=1:m      % pick a single data point
        if (y(numsamples)*dot(x(numsamples,:),w) < 1)  
            w = (1-1/step)*w + 1/(lambda*step)*y(numsamples)*x(numsamples,:)';
        else
            w = (1-1/step)*w;
        end
        step=step+1;         % increment counter
    end
    c1=1-(y.*(x*w));
    c1(c1<0)=0;   
end
  
end 


function [b y] = make_SVM_SGD_FourC(Xd,c1,c2,c3,c4,sizeC,lambda,iter)

v = [c1];
va = repelem(v,sizeC)';

w = [c2];
wa = repelem(w,sizeC)';

u = [c3];
ua = repelem(u,sizeC)';

t = [c4];
ta = repelem(t,sizeC)';

y = vertcat(va,wa,ua,ta);

b = SVM_SGD(Xd,y,lambda,iter); 

figure
d = 0.01; %frequency of x values
 [x1Grid,x2Grid] = meshgrid(0:d:1,0:d:1); %all data points
 xGrid = [x1Grid(:),x2Grid(:)];
 score = predictGrid(xGrid,b(1),b(2),b(3)); 
 %calculate score for all data points
 result = score';
 
gscatter(xGrid(:,1),xGrid(:,2),result,[1 1 0; 0 1 1]);
hold on
gscatter(Xd(:,1),Xd(:,2),y,'br','xs')
hold on;
axis([0 1 0 1]);
xlabel('sensor 3');
ylabel('sensor 6');
legend('Negative Prediction','Positive Prediction','Negative Gas','Positive Gas');
[pl,nl,ml] = makeLines(0,1,b(1),b(2),b(3));

end


function [w] = SVM_SMO(x,y,c,b,kt)
    
Osample=x;
Olabel=y;

N = length(y);
C = c; %Concluded after Cross-Validation
tol = 10e-5;
%tol = 0;
alpha = zeros(N,1); % initializing alphas as zero so that sum of alpha(i)*y(i) = 0 (one of the stationary conditions )
bias = b;
it=0;
maxit=1000;
%  SMO Algorithm
while (it<maxit) % instead of waiting when aplhas can't be changed any more just let it be a fixed iteration
    it = it +1;
    changed_alphas=0;
    %fprintf('changed alphas : %d\n',changed_alphas);
    N=size(y,1);
    for i=1:N % for each alpha(i) loop and pick another alpha chnage it and try to optimize the primal equation following all other constraint and not violating KKT condition
        %fprintf(' i : %d\n',i);
        Ei=sum(alpha.*y.*K(x,x(i,:),kt))-y(i); % calculate the prediction error Ei= f(x(i))-y(i) = w*x(i)-b -y(i)
        % substitue w from stationary equation w= sum(alpha*y*x)
        if ((Ei*y(i)<-tol) && alpha(i)<C)||(Ei*y(i) > tol && (alpha(i) > 0)) % kkt condition is violated 
            %fprintf('kkt condition not met, ei: %d\n',Ei);
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
    if changed_alphas==0
        break
    end
    x=x((find(alpha~=0)),:);
    y=y((find(alpha~=0)),:);
    alpha=alpha((find(alpha~=0)),:);
end
% Weights
W=sum(alpha.*y.*x)
% Bias
bias =-1*mean( y - x*W')

w = [W bias];

% Support Vectors
disp('Number of support Vectors : ')
disp(N)
Xsupport=x;Ysupport=y;

x=Osample;
y=Olabel;

figure;

scatter(x(y==1,1),x(y==1,2),'b');
hold on
scatter(x(y==-1,1),x(y==-1,2),'r');
hold on
scatter(Xsupport(Ysupport==1,1),Xsupport(Ysupport==1,2),'.b');
hold on
scatter(Xsupport(Ysupport==-1,1),Xsupport(Ysupport==-1,2),'.r');
hold on
[plm,nlm,mlm] = makeLines(0,1,W(1),W(2),bias);
plm.LineWidth = 2;
nlm.LineWidth = 2;
mlm.LineWidth = 2;

axis([0 1 0 1])
xlabel('sensor 3');
ylabel('sensor 6');
fprintf('SMO SVM done\n');

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

% Weights
W=sum(alpha.*y.*x);
% Bias
bias =-1*mean( y - x*W');

w = [W bias];

Xsupport=x;Ysupport=y;
x=Osample;
y=Olabel;

%plot SMO SVM

figure;
d = 0.01; %frequency of x values
 [x1Grid,x2Grid] = meshgrid(0:d:1,0:d:1); %all data points
 xGrid = [x1Grid(:),x2Grid(:)];
 score = predictGrid(xGrid,W(1),W(2),bias); 
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
[plm,nlm,mlm] = makeLines(0,1,W(1),W(2),bias);
plm.LineWidth = 2;
nlm.LineWidth = 2;
mlm.LineWidth = 2;

axis([0 1 0 1])
title('SVM - SMO');
legend('Preditced Negative Gas','Predicted Positive Gas','Pos','Neg','Support Vector');
xlabel ('sensor 3');
ylabel ('sensor 6');

end 


function val = predictScore(x,y,w1,w2,w3) % predict for one example
    val = w1*x+w2*y-w3;
    %fprintf('x: %f , y: %f , val = %f\n',x,y,val);
end 

function [val f]= predictGrid(xG,w1,w2,w3) % predict for a grid 
    %val = cell(size(xG));
    %val = zeros(size(xG));
    for k = 1:length(xG)
        %fprintf('%d : ',k)
        r = predictScore(xG(k,1),xG(k,2),w1,w2,w3);
        if (r <0)
            val(k)=0;
            f(k)= r;
        else
            val(k)=1;
            f(k) = r;
        end 
    end 
    
end 

function [J] = sampleMat(mat,inc,sampSize)
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

function [normMat] = normalizeGasData(mat)
    matx = mat(:,2:end);
    N = matx- min(matx);
    De = max(matx)- min(matx);
    normMatx = N./De;
    normMat = [mat(:,1) normMatx]; 
end

function [p,n,m] = makeLines(xl,xh,w1,w2,w3)
    hold on;
lx = linspace(xl,xh);
yp = ((-(1+w3))/(w2))/((1+w3)/(w1))*(lx)+((1+w3)/(w2));
yn = ((-(-1+w3))/(w2))/((-1+w3)/(w1))*(lx)+((-1+w3)/(w2));
ym = (-w1/w2)*(lx)+(w3/w2);
p = plot(lx,yp);
n = plot(lx,yn);
m = plot(lx,ym);

end

function [partMat] = loadData(file,from,to)
data = importdata(file,' ');
partMat = data(from:to,:);
end

function [predictedScore] = scoresOfData(data,w1,w2,w3)
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

function ker=K(Xtrain,x_i,k)

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