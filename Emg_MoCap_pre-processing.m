% M.Sc Thesis- Sprint Study PCA- Process EMG & Kinematic Data to PCA Matrix
% [markerset, 9 Channel EMG] 

clear 
close all 

subject=5; % Change for each trialx
Sprint=2; % Change for each trial 
Height = 1.77; % Change height for each participant
Sex= "M";
MVC=0; % 1= MVC calculator will run, 0= No MVC calculator

% Create Participant Index
for n=1:100
    num=num2str(n,'%03.f'); 
    ParticipantID(:,n)=join(["P",num],'');
end 

Trial={'Trial_1', 'Trial_2','Trial_3','Trial_4'};
Channelname={'GAS','LUM','OBL','BIC','MED','MAX','VLO','REC','LAT'};
Kinfilename={'_Sprint_001.c3d','_Sprint_002.c3d','_Sprint_003.c3d','_Sprint_004.c3d'};

filename='FAST_PCAINPUT_LP250.mat';
foldername='/Users/chris/OneDrive - Brock University/M.Sc/MATLAB/Projects/Sprint Study/Databases';
File=fullfile(foldername,filename);
load(File)

Pidx= find(PCAINPUT.Subj==subject);
VMAXPOINT=PCAINPUT.VMAX(Pidx);
VMAX=max(VMAXPOINT);
idxVMAX=find(VMAX==PCAINPUT.VMAX);
Sprint=PCAINPUT.Sprint(idxVMAX);
%Sprint=4; % Change for each trial 


%% Important Relevent Files 
% Import kinematic files 
foldername=join(["/Users/chris/OneDrive - Brock University/M.Sc/MATLAB/Projects/Sprint Study/Data/",ParticipantID{subject},'/Kinematics/'],'');
kfilename=join([ParticipantID{subject},Kinfilename{Sprint}],'');
kFile=fullfile(foldername,kfilename);
[H, BYTEORDER, STORAGEFORMAT] = btkReadAcquisition(kFile{1});

% 
% Import EMG Data
filename='Full_EMG_Database.mat';
foldername='/Users/chris/OneDrive - Brock University/M.Sc/MATLAB/Projects/Sprint Study/Databases';
File=fullfile(foldername,filename);
load(File)

% load(File) % Load Peaks  
% load('/Users/chris/OneDrive - Brock University/M.Sc/MATLAB/Projects/Sprint Study/Databases/N27_EMG_Database.mat')

%Load PCA database (Add in later) 
filename='FAST_PCAINPUT.mat';
foldername='/Users/chris/OneDrive - Brock University/M.Sc/MATLAB/Projects/Sprint Study/Databases';
File=fullfile(foldername,filename);
load(File)

fs = 60; % Hz
TN_points = 101; 
pts = btkGetPointsValues(H);

% Will need to double check that these don't flip, otherwise the slope
% detection will not work correctly 

x = pts(:,1:3:end-2); % should be x
y = pts(:,2:3:end-1); % should be y
z = pts(:,3:3:end);

LKIN=length(x);

% EMG_All=[EMG.(ParticipantID{subject}).(Trial{Sprint}).GAS,EMG.(ParticipantID{subject}).(Trial{Sprint}).LUM,EMG.(ParticipantID{subject}).(Trial{Sprint}).OBL,EMG.(ParticipantID{subject}).(Trial{Sprint}).BIC,EMG.(ParticipantID{subject}).(Trial{Sprint}).MED,EMG.(ParticipantID{subject}).(Trial{Sprint}).MAX,EMG.(ParticipantID{subject}).(Trial{Sprint}).VLO,EMG.(ParticipantID{subject}).(Trial{Sprint}).REC,EMG.(ParticipantID{subject}).(Trial{Sprint}).LAT];
%% Crop Kinematic and EMG Data 

% Calculate Position of Thorax x,y,z used to determine if the participant
% is progressing in the +x or -x direction 
Thor_x = (x(:,11) + x(:,12) + x(:,13) + x(:,14) + x(:,15) + x(:,21) + x(:,22))./7; 
Thor_y = (y(:,11) + y(:,12) + y(:,13) + y(:,14) + y(:,15) + y(:,21) + y(:,22))./7; 
Thor_z = (z(:,11) + z(:,12) + z(:,13) + z(:,14) + z(:,15) + z(:,21) + z(:,22))./7;

val=sign(x(end));

if val == -1
    % If progrssing in the negative direction 
    x20=min(find(Thor_x<-20000)); %find location of 20m 
    x40=min(find(Thor_x<-40000)); % find location of 40m 
end

if val == 1
    % If progrssing in the positive direction 
    x20=min(find(Thor_x>20000)); %find location of 20m 
    x40=min(find(Thor_x>40000)); % find location of 40m 
end 

mx=(Thor_x(x40)-Thor_x(x20))/(x40-x20); % Calculate slope of x progression
my=(Thor_y(x40)-Thor_y(x20))/(x40-x20); % Calculate slope of y progression 


if mx<0 
    x=x.*-1; % Flip participants x progrssion if progressing negative
    1
end

if my<0 
    y=y.*-1;  % Flip participants y progrssion if progressing negative
    1
end

%% 
G=gray(64)
figure(1) 
plot(x,'LineWidth',2)
hold on 
plot(y,'LineWidth',2)
xlabel('Frames')
ylabel('Distance (mm)')
xlim([0 560])
ylim([0 80000])
set(gca,'FontSize',20)
% set(gcf,'Color','k')
% set(gca,'Color','k')

%% Crop Kinematic and EMG data 
figure(5)
subplot(1,2,1)
plot(x)
xlabel 'frames'
ylabel 'x-displacement (mm)'
title 'SELECT START OF SPRINT'

subplot(1,2,2)
plot(y)
xlabel 'frames'
ylabel 'y-displacement (mm)'
title 'SELECT START OF SPRINT'

[idx,~] = ginput(2);

idx = round(idx);

x = x(idx(1):idx(2),:);
y = y(idx(1):idx(2),:);
z = z(idx(1):idx(2),:);
%%
% EMGidx=idx*33.333; 
% EMG_All=EMG_All(EMGidx(1):EMGidx(2),:);

[~,d] = min(x(1,[53,59]));

if d ==1 % Sets start of the sprint to zero, which is either the right 2 or right toe 
x = x - x(1,53);
y = y - y(1,53);
z = z - z(1,53);
elseif d==2
x = x - x(1,59);
y = y - y(1,59);
z = z - z(1,59);
end

close all

x_all=[]; 
y_all=[];
z_all=[]; 

for n=1:size(x,2)
    x_all=[x_all; x(:,n)];
    y_all=[y_all; y(:,n)];
    z_all=[z_all; z(:,n)];
end

LKIN = size(x,1); 

%% Correct Coordinate System 

x=x_all;
y=y_all; 
z=z_all; 

P=[x y z]; 

x0=x-mean(x) ; y0 = y-mean(y) ; z0=z-mean(z) ; % De-mean the data 

P1 = [x0 y0 z0]; 

scatter3(P1(:,1), P1(:,2), P1(:,3), 'b'); 

HA=[min(P1(:,1)) min(P1(:,2)) max(P1(:,3))+1];%just for better visualaztion
hold on;scatter3(HA(:,1),HA(:,2),HA(:,3),'g.');%just for better visualaztion

PCA=pca(P);
e1=PCA(:,1)'; e2=PCA(:,3)' ;e3=PCA(:,2)';  % 3 principal vector(3 eigenvector) of "input data"
n1=[1 0 0]  ; n2=[0 1 0]   ; n3=[0 0 1];   % 3 unit vector Ox,Oy,Oz

R=[e1;e2;e3];

newdata1=(R*P1')';%new data corresponding to P1 coordinate
hold on; scatter3(newdata1(:,1),newdata1(:,2),newdata1(:,3),'r.');
newdata=[newdata1(:,1)+mean(x),newdata1(:,2)+mean(y),newdata1(:,3)+mean(z)];

x1 = newdata(:,1);
y1 = newdata(:,2);
z1 = newdata(:,3); 


x = reshape(x1, LKIN,64); % New kinematics
y = reshape(y1, LKIN,64); % New kinematics
z= reshape(z1, LKIN,64); % New kinematics 

[~,d] = min(x(1,[53,59]));


if d ==1
x = x - x(1,53);
y = y - y(1,53);
z = z - z(1,53);
elseif d==2
x = x - x(1,59);
y = y - y(1,59);
z = z - z(1,59);
end

figure(3)
subplot(1,3,1)
plot(x)
xlabel 'frames'
ylabel 'x-displacement (mm)'
title 'After'
subplot(1,3,2)
plot(y)
xlabel 'frames'
ylabel 'y-displacement (mm)'
title 'After'
subplot(1,3,3)
plot(z)
xlabel 'frames'
ylabel 'x-displacement (mm)'
title 'After'

%% 
figure(4)
plot(x,'LineWidth',2)
hold on 
plot(y,'LineWidth',2)
xlabel('Frames')
time = (1:size(x,1))./fs;
 
ylabel('Distance (mm)')
xlim([0 560])
ylim([0 80000])
set(gca,'FontSize',20)
%% Estime Approx. Sprint Velocity 
Thor_x = (x(:,11) + x(:,12) + x(:,13) + x(:,14) + x(:,15) + x(:,21) + x(:,22))./7; % recalculate thorax with corrected coordinate system 
Thor_y = (y(:,11) + y(:,12) + y(:,13) + y(:,14) + y(:,15) + y(:,21) + y(:,22))./7; 
Thor_z = (z(:,11) + z(:,12) + z(:,13) + z(:,14) + z(:,15) + z(:,21) + z(:,22))./7; 
Thor_N = sqrt(Thor_x.^2 + Thor_y.^2 + Thor_z.^2);

TimeStep = 1/fs;

Thor_vel = (diff(Thor_N))./TimeStep;
maxVel = max(Thor_vel)./1000;
maxVel = num2str(maxVel);

stop = min(find(Thor_x>60000));
FinishTime = time(stop);
[c,v] = max(Thor_vel);

ThorVel_N_relative = (Thor_vel./max(Thor_vel));
ThorVel_N_relative = abs(ceil(ThorVel_N_relative.*10000));

for j =1:size(x,1)-1
   if ThorVel_N_relative(j,:)==0
       ThorVel_N_relative(j,:)=1;
   end
       
end
%%
n=v;

figure('position',[1 1 1278 642])
hold on
scatter3(x(n,:),y(n,:),z(n,:),10,'filled','k')
%Axial
plot3(x(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),y(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),z(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),'k')
plot3(x(n,[22,11]),y(n,[22,11]),z(n,[22,11]),'k')
plot3(x(n,[21,11]),y(n,[21,11]),z(n,[21,11]),'k')
plot3(x(n,[17,18,19,17,20,18,20,19,20,16]),y(n,[17,18,19,17,20,18,20,19,20,16]),z(n,[17,18,19,17,20,18,20,19,20,16]),'k')

%Right-UE
plot3(x(n,[21,23,24,29,24,21]),y(n,[21,23,24,29,24,21]),z(n,[21,23,24,29,24,21]),'r')
plot3(x(n,[21,29,27,29,28,27]),y(n,[21,29,27,29,28,27]),z(n,[21,29,27,29,28,27]),'r')
plot3(x(n,[24,27,24,28,27]),y(n,[24,27,24,28,27]),z(n,[24,27,24,28,27]),'r')
plot3(x(n,[23,27,23,28,27]),y(n,[23,27,23,28,27]),z(n,[23,27,23,28,27]),'r')
plot3(x(n,[27,33,28,33,34,33,35,33,34,27]),y(n,[27,33,28,33,34,33,35,33,34,27]),z(n,[27,33,28,33,34,33,35,33,34,27]),'r')

%Left-UE
plot3(x(n,[22,25,26,25,32,22]),y(n,[22,25,26,25,32,22]),z(n,[22,25,26,25,32,22]),'g')
plot3(x(n,[22,26]),y(n,[22,26]),z(n,[22,26]),'g')
plot3(x(n,[30,31,25,30,25,31,25,26,30,26,31,26]),y(n,[30,31,25,30,25,31,25,26,30,26,31,26]),z(n,[30,31,25,30,25,31,25,26,30,26,31,26]),'g')
plot3(x(n,[32,30,31,30,32,31,32]),y(n,[32,30,31,30,32,31,32]),z(n,[32,30,31,30,32,31,32]),'g')
plot3(x(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),y(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),z(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),'g')

%Right-LE
plot3(x(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),y(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),z(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),'r')
plot3(x(n,[47,48,47,49]),y(n,[47,48,47,49]),z(n,[47,48,47,49]),'r')
plot3(x(n,[40,48,40,49]),y(n,[40,48,40,49]),z(n,[40,48,40,49]),'r')
plot3(x(n,[41,48,41,49]),y(n,[41,48,41,49]),z(n,[41,48,41,49]),'r')
plot3(x(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),y(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),z(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),'r')

%Left-LE
plot3(x(n,[43,44,43,45,43,46,45,44,46]),y(n,[43,44,43,45,43,46,45,44,46]),z(n,[43,44,43,45,43,46,45,44,46]),'g')
plot3(x(n,[50,46,50,45,50,44,50]),y(n,[50,46,50,45,50,44,50]),z(n,[50,46,50,45,50,44,50]),'g')
plot3(x(n,[50,51,50,52,50]),y(n,[50,51,50,52,50]),z(n,[50,51,50,52,50]),'g')
plot3(x(n,[44,51,44,52,44]),y(n,[44,51,44,52,44]),z(n,[44,51,44,52,44]),'g')
plot3(x(n,[45,51,45,52,45]),y(n,[45,51,45,52,45]),z(n,[45,51,45,52,45]),'g')
plot3(x(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),y(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),z(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),'g')

grid on
axis equal
view(0,0)
zlim ([min(min(z)') max(max(z)')])
xlim ([x(n,11)-1500 x(n,11)+1500])
xlabel 'X (mm)'
ylabel 'Y (mm)'
zlabel 'Z (mm)'
%% Process Data 
% fsEMG=2000;
% LPcutoff=20;
% rec_EMG=abs(EMG_All); 
% [c,d]=butter(2,((LPcutoff/0.802)/(fsEMG/2)),'low');
% LE_EMG=filtfilt(c,d,rec_EMG);
% 
% EMGend=round(min(stop)*33.333);
% 
% EMG_Sprint=LE_EMG(1:EMGend,:);
% 
% % Calculate MVC only if MVC=1 
% if MVC==1
%     x20 = min(find(Thor_x>20000))
%     x20=round(x20*33.333);
%     EMG_20=EMG_Sprint(1:x20,:);
%     figure(5)
%         for n=1:9
%             plot(EMG_20(:,n))
%             [~,P]=ginput(3);
%             peak(:,n)=mean(P,1);
%         end 
%    for n=1:9
%     Norm_Sprint(:,n)=(EMG_Sprint(:,n)./peak(:,n))*100;
%    end
% end
% 
% if MVC== 0 
%     Peak=Data.(ParticipantID{subject}).MVCpeak; 
%     for n=1:9
%         Norm_Sprint(:,n)=(EMG_Sprint(:,n)./Peak(:,n))*100;
%     end
% end

% Crop 5 cycles about max velocity 

figure(4)
plot(z(:,53));
hold on
xline(v,'r','LineWidth',2)

[k,l] = findpeaks(z(:,53)*-1,'MinPeakProminence',150);
scatter(l,k*-1,200,'r.')


closest = abs(v-l); %Max V minus R-R cycles
[~,t] = min(closest);


cycle_index = l(t-3:t+2); %-3 + 2

for n =1:6
xline(cycle_index(n),'k','LineWidtch',1)
end

hold off

% subplot(2,1,2)
% plot(Norm_Sprint(:,1))
% hold on 
% for n=1:6
%     xline(EMGcycle_index(n),'k','LineWidth',1)
% end


Cycle1x = x(cycle_index(1):cycle_index(2),:);
Cycle1y = y(cycle_index(1):cycle_index(2),:);
Cycle1z = z(cycle_index(1):cycle_index(2),:);

Cycle2x = x(cycle_index(2):cycle_index(3),:);
Cycle2y = y(cycle_index(2):cycle_index(3),:);
Cycle2z = z(cycle_index(2):cycle_index(3),:);

Cycle3x = x(cycle_index(3):cycle_index(4),:);
Cycle3y = y(cycle_index(3):cycle_index(4),:);
Cycle3z = z(cycle_index(3):cycle_index(4),:);

Cycle4x = x(cycle_index(4):cycle_index(5),:);
Cycle4y = y(cycle_index(4):cycle_index(5),:);
Cycle4z = z(cycle_index(4):cycle_index(5),:);

Cycle5x = x(cycle_index(5):cycle_index(6),:);
Cycle5y = y(cycle_index(5):cycle_index(6),:);
Cycle5z = z(cycle_index(5):cycle_index(6),:);
% 
% EMGcycle1=Norm_Sprint(EMGcycle_index(1):EMGcycle_index(2),:);
% EMGcycle2=Norm_Sprint(EMGcycle_index(2):EMGcycle_index(3),:);
% EMGcycle3=Norm_Sprint(EMGcycle_index(3):EMGcycle_index(4),:);
% EMGcycle4=Norm_Sprint(EMGcycle_index(4):EMGcycle_index(5),:);
% EMGcycle5=Norm_Sprint(EMGcycle_index(5):EMGcycle_index(6),:);

%% 


%% Time-Normalize Each Cycle for Ensemble
Cycle1x = rubberband(fs,TN_points,Cycle1x); 
Cycle1y = rubberband(fs,TN_points,Cycle1y); 
Cycle1z = rubberband(fs,TN_points,Cycle1z); 

Cycle2x = rubberband(fs,TN_points,Cycle2x); 
Cycle2y = rubberband(fs,TN_points,Cycle2y); 
Cycle2z = rubberband(fs,TN_points,Cycle2z);

Cycle3x = rubberband(fs,TN_points,Cycle3x); 
Cycle3y = rubberband(fs,TN_points,Cycle3y); 
Cycle3z = rubberband(fs,TN_points,Cycle3z);

Cycle4x = rubberband(fs,TN_points,Cycle4x); 
Cycle4y = rubberband(fs,TN_points,Cycle4y); 
Cycle4z = rubberband(fs,TN_points,Cycle4z);

Cycle5x = rubberband(fs,TN_points,Cycle5x); 
Cycle5y = rubberband(fs,TN_points,Cycle5y); 
Cycle5z = rubberband(fs,TN_points,Cycle5z);

% 
% EMGFrames=101;
% EMGcycle1=rubberband(fsEMG,EMGFrames,EMGcycle1);
% EMGcycle2=rubberband(fsEMG,EMGFrames,EMGcycle2);
% EMGcycle3=rubberband(fsEMG,EMGFrames,EMGcycle3);
% EMGcycle4=rubberband(fsEMG,EMGFrames,EMGcycle4);
% EMGcycle5=rubberband(fsEMG,EMGFrames,EMGcycle5);
% 
% figure(4)
% 
% for n=1:9
%     subplot(9,1,n)
%     plot(EMGcycle1(:,n))
%     hold on 
%     plot(EMGcycle2(:,n))
%     plot(EMGcycle3(:,n))
%     plot(EMGcycle4(:,n))
%     plot(EMGcycle5(:,n))
% end

% Set origin as T12 point (Across all frames). 

Cycle1x = Cycle1x - Cycle1x(:,10);
Cycle1y = Cycle1y - Cycle1y(:,10); 
Cycle1z = Cycle1z - Cycle1z(:,10); 

Cycle2x = Cycle2x - Cycle2x(:,10);
Cycle2y = Cycle2y - Cycle2y(:,10); 
Cycle2z = Cycle2z - Cycle2z(:,10);

Cycle3x = Cycle3x - Cycle3x(:,10);
Cycle3y = Cycle3y - Cycle3y(:,10); 
Cycle3z = Cycle3z - Cycle3z(:,10);

Cycle4x = Cycle4x - Cycle4x(:,10);
Cycle4y = Cycle4y - Cycle4y(:,10); 
Cycle4z = Cycle4z - Cycle4z(:,10);

Cycle5x = Cycle5x - Cycle5x(:,10);
Cycle5y = Cycle5y - Cycle5y(:,10); 
Cycle5z = Cycle5z - Cycle5z(:,10);


%%
col=jet(5); 
n=25; 
cidx=1;
figure('units','normalized','outerposition',[0 0 1 1])


x=Cycle1x; 
y=Cycle1y;
z=Cycle1z;

figure(1)
%figure('position',[1 1 1278 642])
hold on
scatter3(x(n,:),y(n,:),z(n,:),10,'filled','Color',col(1,:))
%Axial
plot3(x(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),y(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),z(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),'Color',col(cidx,:))
plot3(x(n,[22,11]),y(n,[22,11]),z(n,[22,11]),'Color',col(cidx,:))
plot3(x(n,[21,11]),y(n,[21,11]),z(n,[21,11]),'Color',col(cidx,:))
plot3(x(n,[17,18,19,17,20,18,20,19,20,16]),y(n,[17,18,19,17,20,18,20,19,20,16]),z(n,[17,18,19,17,20,18,20,19,20,16]),'Color',col(cidx,:))

%Right-UE
plot3(x(n,[21,23,24,29,24,21]),y(n,[21,23,24,29,24,21]),z(n,[21,23,24,29,24,21]),'Color',col(cidx,:))
plot3(x(n,[21,29,27,29,28,27]),y(n,[21,29,27,29,28,27]),z(n,[21,29,27,29,28,27]),'Color',col(cidx,:))
plot3(x(n,[24,27,24,28,27]),y(n,[24,27,24,28,27]),z(n,[24,27,24,28,27]),'Color',col(cidx,:))
plot3(x(n,[23,27,23,28,27]),y(n,[23,27,23,28,27]),z(n,[23,27,23,28,27]),'Color',col(cidx,:))
plot3(x(n,[27,33,28,33,34,33,35,33,34,27]),y(n,[27,33,28,33,34,33,35,33,34,27]),z(n,[27,33,28,33,34,33,35,33,34,27]),'Color',col(cidx,:))

%Left-UE
plot3(x(n,[22,25,26,25,32,22]),y(n,[22,25,26,25,32,22]),z(n,[22,25,26,25,32,22]),'Color',col(cidx,:))
plot3(x(n,[22,26]),y(n,[22,26]),z(n,[22,26]),'Color',col(1,:))
plot3(x(n,[30,31,25,30,25,31,25,26,30,26,31,26]),y(n,[30,31,25,30,25,31,25,26,30,26,31,26]),z(n,[30,31,25,30,25,31,25,26,30,26,31,26]),'Color',col(cidx,:))
plot3(x(n,[32,30,31,30,32,31,32]),y(n,[32,30,31,30,32,31,32]),z(n,[32,30,31,30,32,31,32]),'Color',col(cidx,:))
plot3(x(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),y(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),z(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),'Color',col(cidx,:))

%Right-LE
plot3(x(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),y(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),z(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),'Color',col(cidx,:))
plot3(x(n,[47,48,47,49]),y(n,[47,48,47,49]),z(n,[47,48,47,49]),'Color',col(cidx,:))
plot3(x(n,[40,48,40,49]),y(n,[40,48,40,49]),z(n,[40,48,40,49]),'Color',col(cidx,:))
plot3(x(n,[41,48,41,49]),y(n,[41,48,41,49]),z(n,[41,48,41,49]),'Color',col(cidx,:))
plot3(x(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),y(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),z(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),'Color',col(cidx,:))

%Left-LE
plot3(x(n,[43,44,43,45,43,46,45,44,46]),y(n,[43,44,43,45,43,46,45,44,46]),z(n,[43,44,43,45,43,46,45,44,46]),'Color',col(cidx,:))
plot3(x(n,[50,46,50,45,50,44,50]),y(n,[50,46,50,45,50,44,50]),z(n,[50,46,50,45,50,44,50]),'Color',col(cidx,:))
plot3(x(n,[50,51,50,52,50]),y(n,[50,51,50,52,50]),z(n,[50,51,50,52,50]),'Color',col(cidx,:))
plot3(x(n,[44,51,44,52,44]),y(n,[44,51,44,52,44]),z(n,[44,51,44,52,44]),'Color',col(cidx,:))
plot3(x(n,[45,51,45,52,45]),y(n,[45,51,45,52,45]),z(n,[45,51,45,52,45]),'Color',col(cidx,:))
plot3(x(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),y(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),z(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),'Color',col(cidx,:))


cidx=2;

x=Cycle2x; 
y=Cycle2y;
z=Cycle2z;

hold on
scatter3(x(n,:),y(n,:),z(n,:),10,'filled','Color',col(1,:))
%Axial
plot3(x(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),y(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),z(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),'Color',col(cidx,:))
plot3(x(n,[22,11]),y(n,[22,11]),z(n,[22,11]),'Color',col(cidx,:))
plot3(x(n,[21,11]),y(n,[21,11]),z(n,[21,11]),'Color',col(cidx,:))
plot3(x(n,[17,18,19,17,20,18,20,19,20,16]),y(n,[17,18,19,17,20,18,20,19,20,16]),z(n,[17,18,19,17,20,18,20,19,20,16]),'Color',col(cidx,:))

%Right-UE
plot3(x(n,[21,23,24,29,24,21]),y(n,[21,23,24,29,24,21]),z(n,[21,23,24,29,24,21]),'Color',col(cidx,:))
plot3(x(n,[21,29,27,29,28,27]),y(n,[21,29,27,29,28,27]),z(n,[21,29,27,29,28,27]),'Color',col(cidx,:))
plot3(x(n,[24,27,24,28,27]),y(n,[24,27,24,28,27]),z(n,[24,27,24,28,27]),'Color',col(cidx,:))
plot3(x(n,[23,27,23,28,27]),y(n,[23,27,23,28,27]),z(n,[23,27,23,28,27]),'Color',col(cidx,:))
plot3(x(n,[27,33,28,33,34,33,35,33,34,27]),y(n,[27,33,28,33,34,33,35,33,34,27]),z(n,[27,33,28,33,34,33,35,33,34,27]),'Color',col(cidx,:))

%Left-UE
plot3(x(n,[22,25,26,25,32,22]),y(n,[22,25,26,25,32,22]),z(n,[22,25,26,25,32,22]),'Color',col(cidx,:))
plot3(x(n,[22,26]),y(n,[22,26]),z(n,[22,26]),'Color',col(cidx,:))
plot3(x(n,[30,31,25,30,25,31,25,26,30,26,31,26]),y(n,[30,31,25,30,25,31,25,26,30,26,31,26]),z(n,[30,31,25,30,25,31,25,26,30,26,31,26]),'Color',col(cidx,:))
plot3(x(n,[32,30,31,30,32,31,32]),y(n,[32,30,31,30,32,31,32]),z(n,[32,30,31,30,32,31,32]),'Color',col(cidx,:))
plot3(x(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),y(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),z(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),'Color',col(cidx,:))

%Right-LE
plot3(x(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),y(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),z(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),'Color',col(cidx,:))
plot3(x(n,[47,48,47,49]),y(n,[47,48,47,49]),z(n,[47,48,47,49]),'Color',col(cidx,:))
plot3(x(n,[40,48,40,49]),y(n,[40,48,40,49]),z(n,[40,48,40,49]),'Color',col(cidx,:))
plot3(x(n,[41,48,41,49]),y(n,[41,48,41,49]),z(n,[41,48,41,49]),'Color',col(cidx,:))
plot3(x(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),y(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),z(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),'Color',col(cidx,:))

%Left-LE
plot3(x(n,[43,44,43,45,43,46,45,44,46]),y(n,[43,44,43,45,43,46,45,44,46]),z(n,[43,44,43,45,43,46,45,44,46]),'Color',col(cidx,:))
plot3(x(n,[50,46,50,45,50,44,50]),y(n,[50,46,50,45,50,44,50]),z(n,[50,46,50,45,50,44,50]),'Color',col(cidx,:))
plot3(x(n,[50,51,50,52,50]),y(n,[50,51,50,52,50]),z(n,[50,51,50,52,50]),'Color',col(cidx,:))
plot3(x(n,[44,51,44,52,44]),y(n,[44,51,44,52,44]),z(n,[44,51,44,52,44]),'Color',col(cidx,:))
plot3(x(n,[45,51,45,52,45]),y(n,[45,51,45,52,45]),z(n,[45,51,45,52,45]),'Color',col(cidx,:))
plot3(x(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),y(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),z(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),'Color',col(cidx,:))


cidx=3;
x=Cycle3x; 
y=Cycle3y;
z=Cycle3z;

hold on
scatter3(x(n,:),y(n,:),z(n,:),10,'filled','Color',col(cidx,:))
%Axial
plot3(x(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),y(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),z(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),'Color',col(cidx,:))
plot3(x(n,[22,11]),y(n,[22,11]),z(n,[22,11]),'Color',col(cidx,:))
plot3(x(n,[21,11]),y(n,[21,11]),z(n,[21,11]),'Color',col(cidx,:))
plot3(x(n,[17,18,19,17,20,18,20,19,20,16]),y(n,[17,18,19,17,20,18,20,19,20,16]),z(n,[17,18,19,17,20,18,20,19,20,16]),'Color',col(cidx,:))

%Right-UE
plot3(x(n,[21,23,24,29,24,21]),y(n,[21,23,24,29,24,21]),z(n,[21,23,24,29,24,21]),'Color',col(cidx,:))
plot3(x(n,[21,29,27,29,28,27]),y(n,[21,29,27,29,28,27]),z(n,[21,29,27,29,28,27]),'Color',col(cidx,:))
plot3(x(n,[24,27,24,28,27]),y(n,[24,27,24,28,27]),z(n,[24,27,24,28,27]),'Color',col(cidx,:))
plot3(x(n,[23,27,23,28,27]),y(n,[23,27,23,28,27]),z(n,[23,27,23,28,27]),'Color',col(cidx,:))
plot3(x(n,[27,33,28,33,34,33,35,33,34,27]),y(n,[27,33,28,33,34,33,35,33,34,27]),z(n,[27,33,28,33,34,33,35,33,34,27]),'Color',col(cidx,:))

%Left-UE
plot3(x(n,[22,25,26,25,32,22]),y(n,[22,25,26,25,32,22]),z(n,[22,25,26,25,32,22]),'Color',col(cidx,:))
plot3(x(n,[22,26]),y(n,[22,26]),z(n,[22,26]),'Color',col(cidx,:))
plot3(x(n,[30,31,25,30,25,31,25,26,30,26,31,26]),y(n,[30,31,25,30,25,31,25,26,30,26,31,26]),z(n,[30,31,25,30,25,31,25,26,30,26,31,26]),'Color',col(cidx,:))
plot3(x(n,[32,30,31,30,32,31,32]),y(n,[32,30,31,30,32,31,32]),z(n,[32,30,31,30,32,31,32]),'Color',col(cidx,:))
plot3(x(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),y(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),z(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),'Color',col(cidx,:))

%Right-LE
plot3(x(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),y(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),z(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),'Color',col(cidx,:))
plot3(x(n,[47,48,47,49]),y(n,[47,48,47,49]),z(n,[47,48,47,49]),'Color',col(cidx,:))
plot3(x(n,[40,48,40,49]),y(n,[40,48,40,49]),z(n,[40,48,40,49]),'Color',col(cidx,:))
plot3(x(n,[41,48,41,49]),y(n,[41,48,41,49]),z(n,[41,48,41,49]),'Color',col(cidx,:))
plot3(x(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),y(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),z(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),'Color',col(cidx,:))

%Left-LE
plot3(x(n,[43,44,43,45,43,46,45,44,46]),y(n,[43,44,43,45,43,46,45,44,46]),z(n,[43,44,43,45,43,46,45,44,46]),'Color',col(cidx,:))
plot3(x(n,[50,46,50,45,50,44,50]),y(n,[50,46,50,45,50,44,50]),z(n,[50,46,50,45,50,44,50]),'Color',col(cidx,:))
plot3(x(n,[50,51,50,52,50]),y(n,[50,51,50,52,50]),z(n,[50,51,50,52,50]),'Color',col(cidx,:))
plot3(x(n,[44,51,44,52,44]),y(n,[44,51,44,52,44]),z(n,[44,51,44,52,44]),'Color',col(cidx,:))
plot3(x(n,[45,51,45,52,45]),y(n,[45,51,45,52,45]),z(n,[45,51,45,52,45]),'Color',col(cidx,:))
plot3(x(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),y(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),z(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),'Color',col(cidx,:))

grid on
axis equal
view(0,0)
zlim ([min(min(z)') max(max(z)')])
xlim ([x(n,11)-1500 x(n,11)+1500])
xlabel 'X (mm)'
ylabel 'Y (mm)'
zlabel 'Z (mm)'

cidx=4;
x=Cycle3x; 
y=Cycle3y;
z=Cycle3z;

hold on
scatter3(x(n,:),y(n,:),z(n,:),10,'filled','Color',col(cidx,:))
%Axial
plot3(x(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),y(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),z(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),'Color',col(cidx,:))
plot3(x(n,[22,11]),y(n,[22,11]),z(n,[22,11]),'Color',col(cidx,:))
plot3(x(n,[21,11]),y(n,[21,11]),z(n,[21,11]),'Color',col(cidx,:))
plot3(x(n,[17,18,19,17,20,18,20,19,20,16]),y(n,[17,18,19,17,20,18,20,19,20,16]),z(n,[17,18,19,17,20,18,20,19,20,16]),'Color',col(cidx,:))

%Right-UE
plot3(x(n,[21,23,24,29,24,21]),y(n,[21,23,24,29,24,21]),z(n,[21,23,24,29,24,21]),'Color',col(cidx,:))
plot3(x(n,[21,29,27,29,28,27]),y(n,[21,29,27,29,28,27]),z(n,[21,29,27,29,28,27]),'Color',col(cidx,:))
plot3(x(n,[24,27,24,28,27]),y(n,[24,27,24,28,27]),z(n,[24,27,24,28,27]),'Color',col(cidx,:))
plot3(x(n,[23,27,23,28,27]),y(n,[23,27,23,28,27]),z(n,[23,27,23,28,27]),'Color',col(cidx,:))
plot3(x(n,[27,33,28,33,34,33,35,33,34,27]),y(n,[27,33,28,33,34,33,35,33,34,27]),z(n,[27,33,28,33,34,33,35,33,34,27]),'Color',col(cidx,:))

%Left-UE
plot3(x(n,[22,25,26,25,32,22]),y(n,[22,25,26,25,32,22]),z(n,[22,25,26,25,32,22]),'Color',col(cidx,:))
plot3(x(n,[22,26]),y(n,[22,26]),z(n,[22,26]),'Color',col(cidx,:))
plot3(x(n,[30,31,25,30,25,31,25,26,30,26,31,26]),y(n,[30,31,25,30,25,31,25,26,30,26,31,26]),z(n,[30,31,25,30,25,31,25,26,30,26,31,26]),'Color',col(cidx,:))
plot3(x(n,[32,30,31,30,32,31,32]),y(n,[32,30,31,30,32,31,32]),z(n,[32,30,31,30,32,31,32]),'Color',col(cidx,:))
plot3(x(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),y(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),z(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),'Color',col(cidx,:))

%Right-LE
plot3(x(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),y(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),z(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),'Color',col(cidx,:))
plot3(x(n,[47,48,47,49]),y(n,[47,48,47,49]),z(n,[47,48,47,49]),'Color',col(cidx,:))
plot3(x(n,[40,48,40,49]),y(n,[40,48,40,49]),z(n,[40,48,40,49]),'Color',col(cidx,:))
plot3(x(n,[41,48,41,49]),y(n,[41,48,41,49]),z(n,[41,48,41,49]),'Color',col(cidx,:))
plot3(x(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),y(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),z(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),'Color',col(cidx,:))

%Left-LE
plot3(x(n,[43,44,43,45,43,46,45,44,46]),y(n,[43,44,43,45,43,46,45,44,46]),z(n,[43,44,43,45,43,46,45,44,46]),'Color',col(cidx,:))
plot3(x(n,[50,46,50,45,50,44,50]),y(n,[50,46,50,45,50,44,50]),z(n,[50,46,50,45,50,44,50]),'Color',col(cidx,:))
plot3(x(n,[50,51,50,52,50]),y(n,[50,51,50,52,50]),z(n,[50,51,50,52,50]),'Color',col(cidx,:))
plot3(x(n,[44,51,44,52,44]),y(n,[44,51,44,52,44]),z(n,[44,51,44,52,44]),'Color',col(cidx,:))
plot3(x(n,[45,51,45,52,45]),y(n,[45,51,45,52,45]),z(n,[45,51,45,52,45]),'Color',col(cidx,:))
plot3(x(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),y(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),z(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),'Color',col(cidx,:))

grid on
axis equal
view(0,0)
zlim ([min(min(z)') max(max(z)')])
xlim ([x(n,11)-1500 x(n,11)+1500])
xlabel 'X (mm)'
ylabel 'Y (mm)'
zlabel 'Z (mm)'

cidx=5;
x=Cycle3x; 
y=Cycle3y;
z=Cycle3z;

hold on
scatter3(x(n,:),y(n,:),z(n,:),10,'filled','Color',col(cidx,:))
%Axial
plot3(x(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),y(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),z(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),'Color',col(cidx,:))
plot3(x(n,[22,11]),y(n,[22,11]),z(n,[22,11]),'Color',col(cidx,:))
plot3(x(n,[21,11]),y(n,[21,11]),z(n,[21,11]),'Color',col(cidx,:))
plot3(x(n,[17,18,19,17,20,18,20,19,20,16]),y(n,[17,18,19,17,20,18,20,19,20,16]),z(n,[17,18,19,17,20,18,20,19,20,16]),'Color',col(cidx,:))

%Right-UE
plot3(x(n,[21,23,24,29,24,21]),y(n,[21,23,24,29,24,21]),z(n,[21,23,24,29,24,21]),'Color',col(cidx,:))
plot3(x(n,[21,29,27,29,28,27]),y(n,[21,29,27,29,28,27]),z(n,[21,29,27,29,28,27]),'Color',col(cidx,:))
plot3(x(n,[24,27,24,28,27]),y(n,[24,27,24,28,27]),z(n,[24,27,24,28,27]),'Color',col(cidx,:))
plot3(x(n,[23,27,23,28,27]),y(n,[23,27,23,28,27]),z(n,[23,27,23,28,27]),'Color',col(cidx,:))
plot3(x(n,[27,33,28,33,34,33,35,33,34,27]),y(n,[27,33,28,33,34,33,35,33,34,27]),z(n,[27,33,28,33,34,33,35,33,34,27]),'Color',col(cidx,:))

%Left-UE
plot3(x(n,[22,25,26,25,32,22]),y(n,[22,25,26,25,32,22]),z(n,[22,25,26,25,32,22]),'Color',col(cidx,:))
plot3(x(n,[22,26]),y(n,[22,26]),z(n,[22,26]),'Color',col(cidx,:))
plot3(x(n,[30,31,25,30,25,31,25,26,30,26,31,26]),y(n,[30,31,25,30,25,31,25,26,30,26,31,26]),z(n,[30,31,25,30,25,31,25,26,30,26,31,26]),'Color',col(cidx,:))
plot3(x(n,[32,30,31,30,32,31,32]),y(n,[32,30,31,30,32,31,32]),z(n,[32,30,31,30,32,31,32]),'Color',col(cidx,:))
plot3(x(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),y(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),z(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),'Color',col(cidx,:))

%Right-LE
plot3(x(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),y(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),z(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),'Color',col(cidx,:))
plot3(x(n,[47,48,47,49]),y(n,[47,48,47,49]),z(n,[47,48,47,49]),'Color',col(cidx,:))
plot3(x(n,[40,48,40,49]),y(n,[40,48,40,49]),z(n,[40,48,40,49]),'Color',col(cidx,:))
plot3(x(n,[41,48,41,49]),y(n,[41,48,41,49]),z(n,[41,48,41,49]),'Color',col(cidx,:))
plot3(x(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),y(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),z(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),'Color',col(cidx,:))

%Left-LE
plot3(x(n,[43,44,43,45,43,46,45,44,46]),y(n,[43,44,43,45,43,46,45,44,46]),z(n,[43,44,43,45,43,46,45,44,46]),'Color',col(cidx,:))
plot3(x(n,[50,46,50,45,50,44,50]),y(n,[50,46,50,45,50,44,50]),z(n,[50,46,50,45,50,44,50]),'Color',col(cidx,:))
plot3(x(n,[50,51,50,52,50]),y(n,[50,51,50,52,50]),z(n,[50,51,50,52,50]),'Color',col(cidx,:))
plot3(x(n,[44,51,44,52,44]),y(n,[44,51,44,52,44]),z(n,[44,51,44,52,44]),'Color',col(cidx,:))
plot3(x(n,[45,51,45,52,45]),y(n,[45,51,45,52,45]),z(n,[45,51,45,52,45]),'Color',col(cidx,:))
plot3(x(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),y(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),z(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),'Color',col(cidx,:))

grid on
axis equal
view(0,0)
zlim ([min(min(z)') max(max(z)')])
xlim ([x(n,11)-1500 x(n,11)+1500])
xlabel 'X (mm)'
ylabel 'Y (mm)'
zlabel 'Z (mm)'

%% Take Ensemble Mean
[frames,points] = size(Cycle1x);
col=[0 0 0]

Mean_x = zeros(frames,points);
Mean_y = zeros(frames,points);
Mean_z = zeros(frames,points);

for n =1:points
    Mean_x(:,n) = mean([Cycle1x(:,n),Cycle2x(:,n),Cycle3x(:,n),Cycle4x(:,n),Cycle5x(:,n)],2);
    Mean_y(:,n) = mean([Cycle1y(:,n),Cycle2y(:,n),Cycle3y(:,n),Cycle4y(:,n),Cycle5y(:,n)],2);
    Mean_z(:,n) = mean([Cycle1z(:,n),Cycle2z(:,n),Cycle3z(:,n),Cycle4z(:,n),Cycle5z(:,n)],2);
end
% 
% for n=1:9
%     EMG_Mean(:,n)= mean([EMGcycle1(:,n),EMGcycle2(:,n),EMGcycle3(:,n),EMGcycle4(:,n),EMGcycle5(:,n)],2);
% end
cidx=1; 
x=Mean_x; 
y=Mean_y;
z=Mean_z;

hold on
scatter3(x(n,:),y(n,:),z(n,:),10,'filled','Color',col(cidx,:))
%Axial
plot3(x(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),y(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),z(n,[2,3,5,4,2,6,7,3,7,8,6,8,9,10,11,14,16,21,12,15,11,15,21,15,22,15,12,22,16]),'Color',col(cidx,:))
plot3(x(n,[22,11]),y(n,[22,11]),z(n,[22,11]),'Color',col(cidx,:))
plot3(x(n,[21,11]),y(n,[21,11]),z(n,[21,11]),'Color',col(cidx,:))
plot3(x(n,[17,18,19,17,20,18,20,19,20,16]),y(n,[17,18,19,17,20,18,20,19,20,16]),z(n,[17,18,19,17,20,18,20,19,20,16]),'Color',col(cidx,:))

%Right-UE
plot3(x(n,[21,23,24,29,24,21]),y(n,[21,23,24,29,24,21]),z(n,[21,23,24,29,24,21]),'Color',col(cidx,:))
plot3(x(n,[21,29,27,29,28,27]),y(n,[21,29,27,29,28,27]),z(n,[21,29,27,29,28,27]),'Color',col(cidx,:))
plot3(x(n,[24,27,24,28,27]),y(n,[24,27,24,28,27]),z(n,[24,27,24,28,27]),'Color',col(cidx,:))
plot3(x(n,[23,27,23,28,27]),y(n,[23,27,23,28,27]),z(n,[23,27,23,28,27]),'Color',col(cidx,:))
plot3(x(n,[27,33,28,33,34,33,35,33,34,27]),y(n,[27,33,28,33,34,33,35,33,34,27]),z(n,[27,33,28,33,34,33,35,33,34,27]),'Color',col(cidx,:))

%Left-UE
plot3(x(n,[22,25,26,25,32,22]),y(n,[22,25,26,25,32,22]),z(n,[22,25,26,25,32,22]),'Color',col(cidx,:))
plot3(x(n,[22,26]),y(n,[22,26]),z(n,[22,26]),'Color',col(cidx,:))
plot3(x(n,[30,31,25,30,25,31,25,26,30,26,31,26]),y(n,[30,31,25,30,25,31,25,26,30,26,31,26]),z(n,[30,31,25,30,25,31,25,26,30,26,31,26]),'Color',col(cidx,:))
plot3(x(n,[32,30,31,30,32,31,32]),y(n,[32,30,31,30,32,31,32]),z(n,[32,30,31,30,32,31,32]),'Color',col(cidx,:))
plot3(x(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),y(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),z(n,[30,36,30,37,30,38,36,37,31,36,31,37,31,38,31]),'Color',col(cidx,:))

%Right-LE
plot3(x(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),y(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),z(n,[39,40,39,41,39,42,41,40,42,47,40,47,41,47]),'Color',col(cidx,:))
plot3(x(n,[47,48,47,49]),y(n,[47,48,47,49]),z(n,[47,48,47,49]),'Color',col(cidx,:))
plot3(x(n,[40,48,40,49]),y(n,[40,48,40,49]),z(n,[40,48,40,49]),'Color',col(cidx,:))
plot3(x(n,[41,48,41,49]),y(n,[41,48,41,49]),z(n,[41,48,41,49]),'Color',col(cidx,:))
plot3(x(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),y(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),z(n,[48,49,53,48,53,54,53,55,53,56,57,56,58,53]),'Color',col(cidx,:))

%Left-LE
plot3(x(n,[43,44,43,45,43,46,45,44,46]),y(n,[43,44,43,45,43,46,45,44,46]),z(n,[43,44,43,45,43,46,45,44,46]),'Color',col(cidx,:))
plot3(x(n,[50,46,50,45,50,44,50]),y(n,[50,46,50,45,50,44,50]),z(n,[50,46,50,45,50,44,50]),'Color',col(cidx,:))
plot3(x(n,[50,51,50,52,50]),y(n,[50,51,50,52,50]),z(n,[50,51,50,52,50]),'Color',col(cidx,:))
plot3(x(n,[44,51,44,52,44]),y(n,[44,51,44,52,44]),z(n,[44,51,44,52,44]),'Color',col(cidx,:))
plot3(x(n,[45,51,45,52,45]),y(n,[45,51,45,52,45]),z(n,[45,51,45,52,45]),'Color',col(cidx,:))
plot3(x(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),y(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),z(n,[51,52,51,59,52,59,60,59,60,59,62,59,61,62,63,62,64]),'Color',col(cidx,:))

grid on
axis equal
view(0,0)
zlim ([min(min(z)') max(max(z)')])
xlim ([x(n,11)-1500 x(n,11)+1500])
xlabel 'X (mm)'
ylabel 'Y (mm)'
zlabel 'Z (mm)'

%% Amplidude Normalize (Participant Height)
Mean_x = Mean_x./(Height);
Mean_y = Mean_y./(Height);
Mean_z = Mean_z./(Height);

%% Create participant row for PCA input matrix:
x_all = [];
y_all = [];
z_all = [];

for s = 1:size(Mean_x,2) %<-- combines all data into a single array
    x_all = [x_all; Mean_x(:,s)];
    y_all = [y_all; Mean_y(:,s)];
    z_all = [z_all; Mean_z(:,s)];
end

fullx_all = [];
fully_all = [];
fullz_all = [];


for s = 1:size(x,2) %<-- combines all data into a single array
    fullx_all = [fullx_all; x(:,s)];
    fully_all = [fully_all; y(:,s)];
    fullz_all = [fullz_all; z(:,s)];
end

% EMG_INPUT=[EMG_Mean(:,1)',EMG_Mean(:,2)',EMG_Mean(:,3)',EMG_Mean(:,4)',EMG_Mean(:,5)',EMG_Mean(:,6)',EMG_Mean(:,7)',EMG_Mean(:,8)',EMG_Mean(:,9)'];
FULL=[fullx_all',fully_all', fullz_all'];

INDEX=length(PCAINPUT.Sprint)+1;
%INDEX=15;

PCAINPUT.Sprint(INDEX,:)=Sprint;
PCAINPUT.CycleIndices(INDEX,:)=cycle_index';
PCAINPUT.SPRINTIDX(INDEX,:)=idx;
PCAINPUT.VMAXIDX(INDEX,:)=v;
PCAINPUT.Subj(INDEX,:)=subject;
PCAINPUT.Height(INDEX,:)=Height; 
PCAINPUT.VMAX(INDEX,:)=str2num(maxVel);
%PCAinput.time(INDEX,:)=str2num(FinishTime); 
PCAINPUT.Matrix(INDEX,:)=[x_all',y_all',z_all'];%  
PCAINPUT.x20(INDEX,:)=min(find(Thor_x>20000));
PCAINPUT.LKIN(INDEX,:)=LKIN; 
PCAINPUT.SEX(INDEX,:)=Sex;

Data.(ParticipantID{subject}).(Trial{Sprint}).KIN=FULL;
% Data.(ParticipantID{subject}).(Trial{Sprint}).EMG=EMG_Sprint; 
% Data.(ParticipantID{subject}).(Trial{Sprint}).NEMG=Norm_Sprint; 

if MVC==1
    Data.(ParticipantID{subject}).MVCpeak=peak;
end

filename='FAST_PCAINPUT_LP250.mat';
save(['/Users/chris/OneDrive - Brock University/M.Sc/MATLAB/Projects/Sprint Study/Databases/' filename],'PCAINPUT')
filename='Full_EMG_Database.mat';
save(['/Users/chris/OneDrive - Brock University/M.Sc/MATLAB/Projects/Sprint Study/Databases/' filename],'Data')