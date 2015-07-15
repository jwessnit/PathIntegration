% Path integration network (Haferlach et al.,2007)
% with 3 direction cells
% 
% 

% initialisation
T=5*10^3;                       % length of simulation run (ms)
dt=1;                           % dt is 1 ms

% agent settings
x=0;
y=0;
heading=180;
turn=5;
speed=1;
R=0;
L=0;
DISTANCE=3000;                  % length of the outward path
DISTANCE_TRAVELLED=0;

% direction cells
Dhp=[60;180;300];               % preferred direction of D cells
Dhp=Dhp*pi/180;
cellD=zeros(3,1);
noise=0.001;

% sigmoid neuron parameters
a=0.667;
b_s=-4.372;
os=zeros(1,6);

% memory (CTRNN) neuron parameters
b_c=-1.164;
tau=135518;
c=zeros(1,3);                   % cell potential
om=1./(1+exp(-c+b_c));

% weights
w_Sl=3.974;
w_Sr=3.976;
w_DS=0.719;
w_DM=0.012;
w_MS=3.962;
w_MM=10^-4;

% initialise weight matrices
% connectivity first
C_Sl=[-1;1;-1;1;-1;1];
C_Sr=[1;-1;1;-1;1;-1];
C_DS=[1,1,0,0,0,0;...
      0,0,1,1,0,0;...
      0,0,0,0,1,1];
C_DM=[1,0,0;...
      0,1,0;...
      0,0,1];
C_MS=[0,0,1,0,0,1;...
      0,1,0,0,1,0;...
      1,0,0,1,0,0];
C_MM=[1,0,0;...
      0,1,0;...
      0,0,1];
% now the weight matrices
W_Sl=C_Sl*w_Sl;
W_Sr=C_Sr*w_Sr;
W_DS=C_DS*w_DS;
W_DM=C_DM*w_DM;
W_MS=C_MS*w_MS;
W_MM=C_MM*w_MM;


% traces
trace_firingD=[];
trace_OUT=[];
trace_HOME=[];
trace_cM=[];
trace_sigmoid=[];
trace_motor=[];
trace_IM=[];
trace_IS=[];

counter=0;
update_figure=0;
update_frequency=30;           % every x frames
noise_turn=5;

figure;

fig=figure;
set(fig,'DoubleBuffer','on');
set(gca,'xlim',[-2000 2000],'ylim',[-2000 2000],...
    'NextPlot','replace','Visible','off');
mov = avifile('path_integration.avi');
mov.Quality=100;

for t=1:T

   % plot path
   counter=counter+1;
   update_figure=mod(counter,update_frequency);
   if update_figure==0
      if DISTANCE_TRAVELLED<DISTANCE
         plot(trace_OUT(:,1),trace_OUT(:,2));
         hold on;
         plot(x,y,'b.');
         hold on;
         plot(0,0,'g+');
         title_txt=strcat('Outward path: ',num2str(DISTANCE_TRAVELLED),' of ',num2str(DISTANCE));
         title(title_txt);
      else
         plot(trace_OUT(:,1),trace_OUT(:,2));
         hold on;
         plot(x,y,'g.');
         hold on;
         plot(0,0,'g+');
         hold on;
         plot(trace_HOME(:,1),trace_HOME(:,2),'g');
      end
      axis([-1500 1500 -1500 1500]);
      axis('square');
      F = getframe(gca);
      mov = addframe(mov,F);
   end
   pause(0.005);
   
   % old position coordinates
   x_old=x;
   y_old=y;

   % calculate new position coordinates
   alpha=heading*pi/180;
   x=x+(speed*cos(alpha));
   y=y+(speed*sin(alpha));

   % calculate distance travelled
   dx=x_old-x;
   dy=y_old-y;
   d=sqrt(dx*dx+dy*dy);
   DISTANCE_TRAVELLED=DISTANCE_TRAVELLED+d;

   if DISTANCE_TRAVELLED<DISTANCE
      heading=heading+randn(1)*noise_turn;
      trace_OUT=[trace_OUT;x,y];
   elseif DISTANCE_TRAVELLED>DISTANCE
      if R>L
         heading=heading+turn+randn(1)*noise_turn;
      elseif R<L
         heading=heading-turn+randn(1)*noise_turn;  
      end 
      trace_HOME=[trace_HOME;x,y]; 
   end

   while heading>360
      heading=heading-360;
   end
   while heading<0
      heading=heading+360;
   end

   % compute firing rate of direction cells
   ha=heading*pi/180;   
   cellD(1,1)=dot([cos(Dhp(1,1));sin(Dhp(1,1))],[cos(ha);sin(ha)]...
              +randn(1)*noise);
   cellD(2,1)=dot([cos(Dhp(2,1));sin(Dhp(2,1))],[cos(ha);sin(ha)]...
              +randn(1)*noise);
   cellD(3,1)=dot([cos(Dhp(3,1));sin(Dhp(3,1))],[cos(ha);sin(ha)]...
              +randn(1)*noise);
   trace_firingD=[trace_firingD;transpose(cellD)];
   
   % update memory neurons
   I_D=w_DM*transpose(cellD);
   I_M=W_MM.*repmat(om,3,1);
   IM=I_D+sum(I_M,1);
   c=c+dt/tau*(-c+IM);
   om=1./(1+exp(-c+b_c));
   trace_IM=[trace_IM;IM];
   trace_cM=[trace_cM;om];

   % update sigmoid neurons
   % calculate inputs from direction cells
   I_D=W_DS.*repmat(cellD,1,6);
   IS=sum(I_D,1);
   % calculate inputs from memory neurons
   I_M=W_MS.*repmat(transpose(om),1,6);
   IS=IS+sum(I_M,1);
   % calculate output of sigmoid neurons
   os=1./(1+exp(-a*(IS+b_s)));
   trace_IS=[trace_IS;IS];
   trace_sigmoid=[trace_sigmoid;os];

   % calculate motor outputs
   L=os*W_Sl;
   R=os*W_Sr;
   trace_motor=[trace_motor;L,R];

end


figure;
subplot(2,2,1);
plot(trace_firingD);
title('firing rate D');
subplot(2,2,2);
plot(trace_cM);
title('cell potential M');
subplot(2,2,3);
plot(trace_sigmoid);
title('sigmoid output');
subplot(2,2,4);
plot(trace_motor);
title('motor output');


figure;
subplot(2,1,1);
plot(trace_IM);
title('input IM');
subplot(2,1,2);
plot(trace_IS);
title('input IS');

mov = close(mov);

