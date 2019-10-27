clear
from = 6000;
to = 16000;
diff = to-from;

gA{1} = loadData('.\data1\B1_GEa_F100_R1.txt',from,to);
gA{2} = loadData('.\data1\B1_GEa_F100_R2.txt',from,to);
gA{3} = loadData('.\data1\B1_GEa_F100_R3.txt',from,to);
gA{4} = loadData('.\data1\B1_GEa_F100_R4.txt',from,to);
gA{5} = loadData('.\data1\B2_GEa_F100_R1.txt',from,to);
gA{6} = loadData('.\data1\B2_GEa_F100_R2.txt',from,to);
gA{7} = loadData('.\data1\B2_GEa_F100_R3.txt',from,to);
gA{8} = loadData('.\data1\B2_GEa_F100_R4.txt',from,to);
gA{9} = loadData('.\data1\B1_GEa_F090_R1.txt',from,to);
gA{10} = loadData('.\data1\B1_GEa_F090_R2.txt',from,to);
gA{11} = loadData('.\data1\B1_GEa_F090_R3.txt',from,to);
gA{12} = loadData('.\data1\B1_GEa_F090_R4.txt',from,to);
gA{13} = loadData('.\data1\B2_GEa_F090_R1.txt',from,to);
gA{14} = loadData('.\data1\B2_GEa_F090_R2.txt',from,to);
gA{15} = loadData('.\data1\B2_GEa_F090_R3.txt',from,to);
gA{16} = loadData('.\data1\B2_GEa_F090_R4.txt',from,to);
gA{17} = loadData('.\data1\B1_GEa_F080_R1.txt',from,to);
gA{18} = loadData('.\data1\B1_GEa_F080_R2.txt',from,to);
gA{19} = loadData('.\data1\B1_GEa_F080_R3.txt',from,to);
gA{20} = loadData('.\data1\B1_GEa_F080_R4.txt',from,to);
gA{21} = loadData('.\data1\B2_GEa_F080_R1.txt',from,to);
gA{22} = loadData('.\data1\B2_GEa_F080_R2.txt',from,to);
gA{23} = loadData('.\data1\B2_GEa_F080_R3.txt',from,to);
gA{24} = loadData('.\data1\B2_GEa_F080_R4.txt',from,to);


gB{1} = loadData('.\data1\B1_GCO_F100_R1.txt',from,to);
gB{2} = loadData('.\data1\B1_GCO_F100_R2.txt',from,to);
gB{3} = loadData('.\data1\B1_GCO_F100_R3.txt',from,to);
gB{4} = loadData('.\data1\B1_GCO_F100_R4.txt',from,to);
gB{5} = loadData('.\data1\B2_GCO_F100_R1.txt',from,to);
gB{6} = loadData('.\data1\B2_GCO_F100_R2.txt',from,to);
gB{7} = loadData('.\data1\B2_GCO_F100_R3.txt',from,to);
gB{8} = loadData('.\data1\B2_GCO_F100_R4.txt',from,to);
gB{9} = loadData('.\data1\B1_GCO_F090_R1.txt',from,to);
gB{10} = loadData('.\data1\B1_GCO_F090_R2.txt',from,to);
gB{11} = loadData('.\data1\B1_GCO_F090_R3.txt',from,to);
gB{12} = loadData('.\data1\B1_GCO_F090_R4.txt',from,to);
gB{13} = loadData('.\data1\B2_GCO_F090_R1.txt',from,to);
gB{14} = loadData('.\data1\B2_GCO_F090_R2.txt',from,to);
gB{15} = loadData('.\data1\B2_GCO_F090_R3.txt',from,to);
gB{16} = loadData('.\data1\B2_GCO_F090_R4.txt',from,to);
gB{17} = loadData('.\data1\B1_GCO_F080_R1.txt',from,to);
gB{18} = loadData('.\data1\B1_GCO_F080_R2.txt',from,to);
gB{19} = loadData('.\data1\B1_GCO_F080_R3.txt',from,to);
gB{20} = loadData('.\data1\B1_GCO_F080_R4.txt',from,to);
gB{21} = loadData('.\data1\B2_GCO_F080_R1.txt',from,to);
gB{22} = loadData('.\data1\B2_GCO_F080_R2.txt',from,to);
gB{23} = loadData('.\data1\B2_GCO_F080_R3.txt',from,to);
gB{24} = loadData('.\data1\B2_GCO_F080_R4.txt',from,to);

gC{1} = loadData('.\data1\B1_GEy_F100_R1.txt',from,to);
gC{2} = loadData('.\data1\B1_GEy_F100_R2.txt',from,to);
gC{3} = loadData('.\data1\B1_GEy_F100_R3.txt',from,to);
gC{4} = loadData('.\data1\B1_GEy_F100_R4.txt',from,to);
gC{5} = loadData('.\data1\B2_GEy_F100_R1.txt',from,to);
gC{6} = loadData('.\data1\B2_GEy_F100_R2.txt',from,to);
gC{7} = loadData('.\data1\B2_GEy_F100_R3.txt',from,to);
gC{8} = loadData('.\data1\B2_GEy_F100_R4.txt',from,to);
gC{9} = loadData('.\data1\B1_GEy_F090_R1.txt',from,to);
gC{10} = loadData('.\data1\B1_GEy_F090_R2.txt',from,to);
gC{11} = loadData('.\data1\B1_GEy_F090_R3.txt',from,to);
gC{12} = loadData('.\data1\B1_GEy_F090_R4.txt',from,to);
gC{13} = loadData('.\data1\B2_GEy_F090_R1.txt',from,to);
gC{14} = loadData('.\data1\B2_GEy_F090_R2.txt',from,to);
gC{15} = loadData('.\data1\B2_GEy_F090_R3.txt',from,to);
gC{16} = loadData('.\data1\B2_GEy_F090_R4.txt',from,to);
gC{17} = loadData('.\data1\B1_GEy_F080_R1.txt',from,to);
gC{18} = loadData('.\data1\B1_GEy_F080_R2.txt',from,to);
gC{19} = loadData('.\data1\B1_GEy_F080_R3.txt',from,to);
gC{20} = loadData('.\data1\B1_GEy_F080_R4.txt',from,to);
gC{21} = loadData('.\data1\B2_GEy_F080_R1.txt',from,to);
gC{22} = loadData('.\data1\B2_GEy_F080_R2.txt',from,to);
gC{23} = loadData('.\data1\B2_GEy_F080_R3.txt',from,to);
gC{24} = loadData('.\data1\B2_GEy_F080_R4.txt',from,to);

gD{1} = loadData('.\data1\B1_GMe_F100_R1.txt',from,to);
gD{2} = loadData('.\data1\B1_GMe_F100_R2.txt',from,to);
gD{3} = loadData('.\data1\B1_GMe_F100_R3.txt',from,to);
gD{4} = loadData('.\data1\B1_GMe_F100_R4.txt',from,to);
gD{5} = loadData('.\data1\B2_GMe_F100_R1.txt',from,to);
gD{6} = loadData('.\data1\B2_GMe_F100_R2.txt',from,to);
gD{7} = loadData('.\data1\B2_GMe_F100_R3.txt',from,to);
gD{8} = loadData('.\data1\B2_GMe_F100_R4.txt',from,to);
gD{9} = loadData('.\data1\B1_GMe_F090_R1.txt',from,to);
gD{10} = loadData('.\data1\B1_GMe_F090_R2.txt',from,to);
gD{11} = loadData('.\data1\B1_GMe_F090_R3.txt',from,to);
gD{12} = loadData('.\data1\B1_GMe_F090_R4.txt',from,to);
gD{13} = loadData('.\data1\B2_GMe_F090_R1.txt',from,to);
gD{14} = loadData('.\data1\B2_GMe_F090_R2.txt',from,to);
gD{15} = loadData('.\data1\B2_GMe_F090_R3.txt',from,to);
gD{16} = loadData('.\data1\B2_GMe_F090_R4.txt',from,to);
gD{17} = loadData('.\data1\B1_GMe_F080_R1.txt',from,to);
gD{18} = loadData('.\data1\B1_GMe_F080_R2.txt',from,to);
gD{19} = loadData('.\data1\B1_GMe_F080_R3.txt',from,to);
gD{20} = loadData('.\data1\B1_GMe_F080_R4.txt',from,to);
gD{21} = loadData('.\data1\B2_GMe_F080_R1.txt',from,to);
gD{22} = loadData('.\data1\B2_GMe_F080_R2.txt',from,to);
gD{23} = loadData('.\data1\B2_GMe_F080_R3.txt',from,to);
gD{24} = loadData('.\data1\B2_GMe_F080_R4.txt',from,to);

gH{1} = loadData('.\data1\B1_GMe_F010_R1.txt',from,to);
gH{2} = loadData('.\data1\B1_GMe_F010_R2.txt',from,to);
gH{3} = loadData('.\data1\B1_GMe_F010_R3.txt',from,to);
gH{4} = loadData('.\data1\B1_GMe_F010_R4.txt',from,to);
gH{5} = loadData('.\data1\B2_GMe_F010_R1.txt',from,to);
gH{6} = loadData('.\data1\B2_GMe_F010_R2.txt',from,to);
gH{7} = loadData('.\data1\B2_GMe_F010_R3.txt',from,to);
gH{8} = loadData('.\data1\B2_GMe_F010_R4.txt',from,to);
gH{9} = loadData('.\data1\B1_GMe_F020_R1.txt',from,to);
gH{10} = loadData('.\data1\B1_GMe_F020_R2.txt',from,to);
gH{11} = loadData('.\data1\B1_GMe_F020_R3.txt',from,to);
gH{12} = loadData('.\data1\B1_GMe_F020_R4.txt',from,to);
gH{13} = loadData('.\data1\B2_GMe_F020_R1.txt',from,to);
gH{14} = loadData('.\data1\B2_GMe_F020_R2.txt',from,to);
gH{15} = loadData('.\data1\B2_GMe_F020_R3.txt',from,to);
gH{16} = loadData('.\data1\B2_GMe_F020_R4.txt',from,to);
gH{17} = loadData('.\data1\B1_GMe_F030_R1.txt',from,to);
gH{18} = loadData('.\data1\B1_GMe_F030_R2.txt',from,to);
gH{19} = loadData('.\data1\B1_GMe_F030_R3.txt',from,to);
gH{20} = loadData('.\data1\B1_GMe_F030_R4.txt',from,to);
gH{21} = loadData('.\data1\B2_GMe_F030_R1.txt',from,to);
gH{22} = loadData('.\data1\B2_GMe_F030_R2.txt',from,to);
gH{23} = loadData('.\data1\B2_GMe_F030_R3.txt',from,to);
gH{24} = loadData('.\data1\B2_GMe_F030_R4.txt',from,to);

szgasA = size(gA(1,1:end));
szgasB = size(gB(1,1:end));
szgasC = size(gC(1,1:end));
szgasD = size(gD(1,1:end));
szgasH = size(gH(1,1:end));

highPPMGas = vertcat(gA{1,1:end},gB{1,1:end},gC{1,1:end},gD{1,1:end});
lowPPMGas = vertcat(gH{1,1:end});

normHighPPM = normalizeGasData(highPPMGas);
normLowPPM = normalizeGasData(lowPPMGas);


numOfSamp = 5;
freq = diff/numOfSamp;

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

for(i=0:szgasH(2)-1)
    sLowgD{i+1} = sampleMat(normLowPPM((i)*diff+1:(i+1)*diff,:),freq,numOfSamp);
end

allS = vertcat(sHighgA{1,1:end},sHighgB{1,1:end},sHighgC{1,1:end},sHighgD{1,1:end},sLowgD{1,1:end});

szAll = size(allS);
szA = [szAll(1)/5 9];
szB = [szAll(1)/5 9];
szC = [szAll(1)/5 9];
szD = [szAll(1)/5 9];
szH = [szAll(1)/5 9];

sA = 7;
sB = 9;
sC = 3;

figure;
scatter3(allS(1:szA(1),sA),allS(1:szA(1),sB),allS(1:szA(1),sC),10,[0 0 1]);
hold on;
scatter3(allS(szA(1)+1:szA(1)+szB(1),sA),allS(szA(1)+1:szA(1)+szB(1),sB),allS(szA(1)+1:szA(1)+szB(1),sC),10,[0 0.5 0]);
hold on;
scatter3(allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sA),allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sB), allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sC),10,[0.75 0.75 0]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sA),allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sB), allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sC),10,[0 0.75 0.75]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sA),allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sB), allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sC),10,[0.75 0 0.75]);
hold on;

X(1:szAll(1),1:4)=0;

for (j=1:szAll(1))
    X(j,1) = allS(j,sA);
    X(j,2) = allS(j,sB);
    X(j,3) = allS(j,sC);
    X(j,4) = -1;
end

[nsamples,nfeatures] = size(X);

%------------------SVM 1 -----------

v = [1];
va = repelem(v,szA(1))';

w = [-1];
wa = repelem(w,szB(1))';

u = [-1];
ua = repelem(u,szC(1))';

t = [-1];
ta = repelem(t,szD(1))';

h = [-1];
ha = repelem(h,szH(1))';

y1 = vertcat(va,wa,ua,ta,ha);

b1 = SVM(X,y1,0.01,10000); % iteration more than 10 times the training set as per Andrew Ng recommendation 

d = 0.05;
 [x1Grid,x2Grid,x3Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
 min(X(:,2)):d:max(X(:,2)), min(X(:,3)):d:max(X(:,3)));
 xGrid = [x1Grid(:),x2Grid(:),x3Grid(:)];
 size(xGrid);
 N = size(xGrid,1);
 [score1,pos1,neg1] = predictGrid(xGrid,b1(1),b1(2),b1(3),b1(4));
 result1 = score1';
 
figure;
scatter3(allS(1:szA(1),sA),allS(1:szA(1),sB),allS(1:szA(1),sC),10,[0 0 1]);
hold on;
scatter3(allS(szA(1)+1:szA(1)+szB(1),sA),allS(szA(1)+1:szA(1)+szB(1),sB),allS(szA(1)+1:szA(1)+szB(1),sC),10,[0 0.5 0]);
hold on;
scatter3(allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sA),allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sB), allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sC),10,[0.75 0.75 0]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sA),allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sB), allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sC),10,[0 0.75 0.75]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sA),allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sB), allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sC),10,[0.75 0 0.75]);
hold on;

if (size(pos1)>0)
    scatter3(pos1(:,1),pos1(:,2),pos1(:,3),10,[0.9290, 0.6940, 0.1250]);
end
hold on;
if (size(neg1)>0)
    scatter3(neg1(:,1),neg1(:,2),neg1(:,3),10,[0 1 1]);
end
 
 %------------------SVM 2 -----------

v = [-1];
va = repelem(v,szA(1))';

w = [1];
wa = repelem(w,szB(1))';

u = [-1];
ua = repelem(u,szC(1))';

t = [-1];
ta = repelem(t,szD(1))';

h = [-1];
ha = repelem(h,szH(1))';

y2 = vertcat(va,wa,ua,ta,ha);

b2 = SVM(X,y2,0.01,10000); % iteration more than 10 times the training set as per Andrew Ng recommendation 

d = 0.05;
 [x1Grid,x2Grid,x3Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
 min(X(:,2)):d:max(X(:,2)), min(X(:,3)):d:max(X(:,3)));
 xGrid = [x1Grid(:),x2Grid(:),x3Grid(:)];
 size(xGrid);
 N = size(xGrid,1);
 [score2,pos2,neg2] = predictGrid(xGrid,b2(1),b2(2),b2(3),b2(4));
 result2 = score2';
  
figure;
scatter3(allS(1:szA(1),sA),allS(1:szA(1),sB),allS(1:szA(1),sC),10,[0 0 1]);
hold on;
scatter3(allS(szA(1)+1:szA(1)+szB(1),sA),allS(szA(1)+1:szA(1)+szB(1),sB),allS(szA(1)+1:szA(1)+szB(1),sC),10,[0 0.5 0]);
hold on;
scatter3(allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sA),allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sB), allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sC),10,[0.75 0.75 0]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sA),allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sB), allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sC),10,[0 0.75 0.75]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sA),allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sB), allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sC),10,[0.75 0 0.75]);
hold on;

if (size(pos2)>0)
    scatter3(pos2(:,1),pos2(:,2),pos2(:,3),10,[0.9290, 0.6940, 0.1250]);
end
hold on;
if (size(neg2)>0)
    scatter3(neg2(:,1),neg2(:,2),neg2(:,3),10,[0 1 1]);
end
 %------------------SVM 3 -----------

v = [-1];
va = repelem(v,szA(1))';

w = [-1];
wa = repelem(w,szB(1))';

u = [1];
ua = repelem(u,szC(1))';

t = [-1];
ta = repelem(t,szD(1))';

h = [-1];
ha = repelem(h,szH(1))';

y3 = vertcat(va,wa,ua,ta,ha);

b3 = SVM(X,y3,0.01,10000); % iteration more than 10 times the training set as per Andrew Ng recommendation 

d = 0.05;
 [x1Grid,x2Grid,x3Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
 min(X(:,2)):d:max(X(:,2)), min(X(:,3)):d:max(X(:,3)));
 xGrid = [x1Grid(:),x2Grid(:),x3Grid(:)];
 size(xGrid);
 N = size(xGrid,1);
 [score3,pos3,neg3] = predictGrid(xGrid,b3(1),b3(2),b3(3),b3(4));
 result3 = score3';
  
figure;
scatter3(allS(1:szA(1),sA),allS(1:szA(1),sB),allS(1:szA(1),sC),10,[0 0 1]);
hold on;
scatter3(allS(szA(1)+1:szA(1)+szB(1),sA),allS(szA(1)+1:szA(1)+szB(1),sB),allS(szA(1)+1:szA(1)+szB(1),sC),10,[0 0.5 0]);
hold on;
scatter3(allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sA),allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sB), allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sC),10,[0.75 0.75 0]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sA),allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sB), allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sC),10,[0 0.75 0.75]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sA),allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sB), allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sC),10,[0.75 0 0.75]);
hold on;

if (size(pos3)>0)
    scatter3(pos3(:,1),pos3(:,2),pos3(:,3),10,[0.9290, 0.6940, 0.1250]);
end
hold on;
if (size(neg3)>0)
    scatter3(neg3(:,1),neg3(:,2),neg3(:,3),10,[0 1 1]);
end
 
 %------------------SVM4 -----------

v = [-1];
va = repelem(v,szA(1))';

w = [-1];
wa = repelem(w,szB(1))';

u = [-1];
ua = repelem(u,szC(1))';

t = [1];
ta = repelem(t,szD(1))';

h = [-1];
ha = repelem(h,szH(1))';

y4 = vertcat(va,wa,ua,ta,ha);

b4 = SVM(X,y4,0.01,10000); % iteration more than 10 times the training set as per Andrew Ng recommendation 

d = 0.05;
 [x1Grid,x2Grid,x3Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
 min(X(:,2)):d:max(X(:,2)), min(X(:,3)):d:max(X(:,3)));
 xGrid = [x1Grid(:),x2Grid(:),x3Grid(:)];
 size(xGrid);
 N = size(xGrid,1);
 [score4,pos4,neg4] = predictGrid(xGrid,b4(1),b4(2),b4(3),b4(4));
 result4 = score4';
 
  
figure;
scatter3(allS(1:szA(1),sA),allS(1:szA(1),sB),allS(1:szA(1),sC),10,[0 0 1]);
hold on;
scatter3(allS(szA(1)+1:szA(1)+szB(1),sA),allS(szA(1)+1:szA(1)+szB(1),sB),allS(szA(1)+1:szA(1)+szB(1),sC),10,[0 0.5 0]);
hold on;
scatter3(allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sA),allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sB), allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sC),10,[0.75 0.75 0]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sA),allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sB), allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sC),10,[0 0.75 0.75]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sA),allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sB), allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sC),10,[0.75 0 0.75]);
hold on;

if (size(pos4)>0)
    scatter3(pos4(:,1),pos4(:,2),pos4(:,3),10,[0.9290, 0.6940, 0.1250]);
end
hold on;
if (size(neg4)>0)
    scatter3(neg4(:,1),neg4(:,2),neg4(:,3),10,[0 1 1]);
end
 %------------------SVM 5 -----------

v = [-1];
va = repelem(v,szA(1))';

w = [-1];
wa = repelem(w,szB(1))';

u = [-1];
ua = repelem(u,szC(1))';

t = [-1];
ta = repelem(t,szD(1))';

h = [1];
ha = repelem(h,szH(1))';

y5 = vertcat(va,wa,ua,ta,ha);

b5 = SVM(X,y5,0.01,10000); % iteration more than 10 times the training set as per Andrew Ng recommendation 

d = 0.05;
 [x1Grid,x2Grid,x3Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
 min(X(:,2)):d:max(X(:,2)), min(X(:,3)):d:max(X(:,3)));
 xGrid = [x1Grid(:),x2Grid(:),x3Grid(:)];
 size(xGrid);
 N = size(xGrid,1);
 [score5,pos5,neg5] = predictGrid(xGrid,b5(1),b5(2),b5(3),b5(4));
 result5 = score5';
 
  
figure;
scatter3(allS(1:szA(1),sA),allS(1:szA(1),sB),allS(1:szA(1),sC),10,[0 0 1]);
hold on;
scatter3(allS(szA(1)+1:szA(1)+szB(1),sA),allS(szA(1)+1:szA(1)+szB(1),sB),allS(szA(1)+1:szA(1)+szB(1),sC),10,[0 0.5 0]);
hold on;
scatter3(allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sA),allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sB), allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sC),10,[0.75 0.75 0]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sA),allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sB), allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sC),10,[0 0.75 0.75]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sA),allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sB), allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sC),10,[0.75 0 0.75]);
hold on;

if (size(pos2)>0)
    scatter3(pos5(:,1),pos5(:,2),pos5(:,3),10,[0.9290, 0.6940, 0.1250]);
end
hold on;
if (size(neg5)>0)
    scatter3(neg5(:,1),neg5(:,2),neg5(:,3),10,[0 1 1]);
end
 %-------------------------------------------------------
 
 %d = 0.015;
 d = 0.025;
 [x1Grid,x2Grid,x3Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
 min(X(:,2)):d:max(X(:,2)), min(X(:,3)):d:max(X(:,3)));
 xGrid = [x1Grid(:),x2Grid(:),x3Grid(:)];
 
 N = size(xGrid,1);
 Scores = zeros(N,5);
 
 [label,score] = predictGridX(xGrid,b1(1),b1(2),b1(3),b1(4));
 Scores(:,1) = score; 
 [label,score] = predictGridX(xGrid,b2(1),b2(2),b2(3),b2(4));
 Scores(:,2) = score; 
 [label,score] = predictGridX(xGrid,b3(1),b3(2),b3(3),b3(4));
 Scores(:,3) = score; 
 [label,score] = predictGridX(xGrid,b4(1),b4(2),b4(3),b4(4));
 Scores(:,4) = score; 
 [label,score] = predictGridX(xGrid,b5(1),b5(2),b5(3),b5(4));
 Scores(:,5) = score; 
 
 
 [~,type] = max(Scores,[],2);

 color1 = zeros(1,3);
 color1c = 1;
 color2 = zeros(1,3);
 color2c = 1;
 color3 = zeros(1,3);
 color3c = 1;
 color4 = zeros(1,3);
 color4c = 1;
 color5 = zeros(1,3);
 color5c = 1;
 
 
 szxGrid = size(xGrid);
 szType = size(type);
 for(s=1:szType(1))
    if(type(s,1)==1)
        color1(color1c,1)=xGrid(s,1);
        color1(color1c,2)=xGrid(s,2);
        color1(color1c,3)=xGrid(s,3);
        color1c = color1c + 1;
    elseif(type(s,1)==2)
        color2(color2c,1)=xGrid(s,1);
        color2(color2c,2)=xGrid(s,2);
        color2(color2c,3)=xGrid(s,3);
        color2c = color2c + 1;
    elseif(type(s,1)==3)
        color3(color3c,1)=xGrid(s,1);
        color3(color3c,2)=xGrid(s,2);
        color3(color3c,3)=xGrid(s,3);
        color3c = color3c + 1;
    elseif(type(s,1)==4)
        color4(color4c,1)=xGrid(s,1);
        color4(color4c,2)=xGrid(s,2);
        color4(color4c,3)=xGrid(s,3);
        color4c = color4c + 1;
    elseif(type(s,1)==5)
        color5(color5c,1)=xGrid(s,1);
        color5(color5c,2)=xGrid(s,2);
        color5(color5c,3)=xGrid(s,3);
        color5c = color5c + 1;
    end
 end
 
figure;
scatter3(color1(:,1),color1(:,2),color1(:,3),10,[0 0 1]);
hold on;
scatter3(color2(:,1),color2(:,2),color2(:,3),10,[0 0.5 0]);
hold on;
scatter3(color3(:,1),color3(:,2),color3(:,3),10,[0.75 0.75 0]);
hold on;
scatter3(color4(:,1),color4(:,2),color4(:,3),10,[0 0.75 0.75]);
hold on;
scatter3(color5(:,1),color5(:,2),color5(:,3),10,[0.75 0 0.75]);
hold on;
scatter3(allS(1:szA(1),sA),allS(1:szA(1),sB),allS(1:szA(1),sC),20,[0.75 0.75 0]);
hold on;
scatter3(allS(szA(1)+1:szA(1)+szB(1),sA),allS(szA(1)+1:szA(1)+szB(1),sB),allS(szA(1)+1:szA(1)+szB(1),sC),20,[0.9290 0.6940 0.1250]);
hold on;
scatter3(allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sA),allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sB), allS(szA(1)+szB(1)+1:szA(1)+szB(1)+szC(1),sC),20,[0.4940, 0.1840, 0.5560]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sA),allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sB), allS(szA(1)+szB(1)+szC(1)+1:szA(1)+szB(1)+szC(1)+szD(1),sC),20,[0.6350, 0.0780, 0.1840]);
hold on;
scatter3(allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sA),allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sB), allS(szA(1)+szB(1)+szC(1)+szD(1)+1:szA(1)+szB(1)+szC(1)+szD(1)+szH(1),sC),20,[0.3010, 0.7450, 0.9330]);
hold on;

title('{\bf Gas Classification Regions}');

xlabel('Sensor 2'); % column 2 in data file

ylabel('Sensor 4'); %column 5 in data file

zlabel('Sensor 7');

axis tight
 
 
 
 %------------------------------------------------------
function [w] = SVM(x,y,lambda,iterations)
[m,d] = size(x);
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
        step=step+1;         % increment counter
    end
    c1=1-(y.*(x*w));
    c1(c1<0)=0;  
    
    
end
  
end 


function [partMat] = loadData(file,from,to)
data = importdata(file,' ');
partMat = data(from:to,:);
end

function val = predict(x,y,z,w1,w2,w3,w4) % predict for one example
    val = w1*x+w2*y+w3*z-w4;
end 

function [val po ne]= predictGrid(xG,w1,w2,w3,w4) % predict for a grid 
  
    po= [];
    ne=[];
    for k = 1:length(xG)
       
        r = predict(xG(k,1),xG(k,2),xG(k,3),w1,w2,w3,w4);
        if (r <0)
            val(k)=0;
            ne = vertcat(ne , [xG(k,1) xG(k,2) xG(k,3)]);
         else
            val(k)=1;
            po = vertcat(po , [xG(k,1) xG(k,2) xG(k,3)]);
        end 
    end 
    
end 

function [val f]= predictGridX(xG,w1,w2,w3,w4) % predict for a grid 
    for k = 1:length(xG)
        r = predict(xG(k,1),xG(k,2),xG(k,3),w1,w2,w3,w4);
        if (r <0)
            val(k)=0;
            f(k)= r;
        else
            val(k)=1;
            f(k) = r;
        end 
    end 
    
end 

function [normMat] = normalizeGasData(mat)
    matx = mat(:,2:end);
    N = matx- min(matx);
    De = max(matx)- min(matx);
    normMatx = N./De;
    normMat = [mat(:,1) normMatx]; 
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