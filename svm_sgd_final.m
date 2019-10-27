% training data X along with bias (set to 1)
X = [-2 4 -1;4 1 -1;1 6 -1;2 4 -1;6 2 -1;2 2 -1;-1 1.5 -1; 1 3 -1;5 2 -1;4 3 -1;4.1 3 -1;4.2 3.1 -1;1 2.5 -1; 0 4 -1;1 5 -1]
% label data y 
y = [-1; -1; 1; 1; 1; -1; -1; -1; 1; 1; 1 ; 1; -1 ; -1; 1]

%figure;
%gscatter(X(:,1),X(:,2),y,'br','xo')
%axis([-7 7 -7 7]);
%line([-2 6], [6 0.5]); 

b = SVM(X,y,0.00001,10000); % iteration more than 10 times the training set as per Andrew Ng recommendation 
fprintf('%f %f %f\n', b.')

[nsamples,nfeatures] = size(X);

for (j=1:nsamples)
   predict(X(j,1),X(j,2),b(1),b(2),b(3));
end

 d = 0.02;
 [x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
 min(X(:,2)):d:max(X(:,2)));
 xGrid = [x1Grid(:),x2Grid(:)];
 size(xGrid);
 N = size(xGrid,1);
 min(X(:,1))
 max(X(:,1))
 min(X(:,2))
 max(X(:,2))
 score = predictGrid(xGrid,b(1),b(2),b(3));
 result = score';
 
figure
h(1:2)= gscatter(xGrid(:,1),xGrid(:,2),result,...
    [1 1 0; 0 1 1]);
hold on
h(3:4)= gscatter(X(:,1),X(:,2),y,'br','xo')
title('{\bf Classification Regions}');
xlabel('feature 1');
ylabel('feature 2');
legend(h,{'negative region','positive region', 'negative sample', 'positive sample'},'Location','Northwest');
axis tight
hold off

function [w] = SVM(x,y,lambda,iterations)
[m,d] = size(x);
fprintf('size of training data: %d\n', m);
w = zeros(d,1);
step = 1;
for i=1:iterations      % iterations over the full data set as per stochastic gradient descent algorithm
    x2 = [x y];
    %shuffle the data 
    random_x = x2(randperm(size(x2, 1)), :);
    x = random_x(:,1:end-1);
    y = random_x(:,end);
    for numsamples=1:m      % pick a single data point
        if (y(numsamples)*dot(x(numsamples,:),w) < 1)  
            w = (1-1/step)*w + 1/(lambda*step)*y(numsamples)*x(numsamples,:)';
            %fprintf('y(numsamples)=%d\n'y(numsamples));
        else
            w = (1-1/step)*w;
        end
        %fprintf('iter=%d step=%d numsamples=%d \n' , i,t,numsamples);
        %y(numsamples)
        %x(numsamples,:)
        %y(numsamples)*x(numsamples,:)'
        %1/(lambda*t)*y(numsamples)*x(numsamples,:)'

        step=step+1;         % increment counter
    end
    c1=1-(y.*(x*w));
    c1(c1<0)=0;  
    cost = w'*w+ lambda*(sum(c1));
    %fprintf('Cost:%f\n',cost);
    
    
end
  
end 

function val = predict(x,y,w1,w2,w3) % predict for one example
    val = w1*x+w2*y-w3;
    fprintf('x: %f , y: %f , val = %f\n',x,y,val);
end 

function val= predictGrid(xG,w1,w2,w3) % predict for a grid 
    %val = cell(size(xG));
    %val = zeros(size(xG));
    for k = 1:length(xG)
        fprintf('%d : ',k)
        r = predict(xG(k,1),xG(k,2),w1,w2,w3);
        if (r <0)
            val(k)=0;
        else
            val(k)=1;
        end 
    end 
    
end 