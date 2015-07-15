% Path integration network (Haferlach et al.,2007)
% with 3 direction cells
% 
% 

% initialisation
T=10*10^3;                      % length of simulation run
dt=1;                           % dt is 1

% agent settings
x=0;
y=0;
heading=180;
turn=1;
speed=1;
OUT=1;                          % visit 3 waypoints and set OUT=0
HOME=0;                         % and HOME=1 when home and WAY=3
WAY=1;
R=0;
L=0;
x_w1=1000;
y_w1=-1000;
x_w2=2000;
y_w2=0;
x_w3=-1000;
y_w3=0;
DISTANCE=5;

% direction cells
Dhp=[60;180;300];               % preferred direction of D cells
Dhp=Dhp*pi/180;
cellD=zeros(3,1);
noise=0.05;

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


%while OUT==0 & HOME==1
for t=1:T

   if WAY>2 & sum(om,2)==0
      speed=0;
   end

   alpha=heading*pi/180;
   x=x+(speed*cos(alpha));
   y=y+(speed*sin(alpha));

   % calculate heading towards waypoint
   if WAY<4 & WAY==1
      x_w=x_w1;
      y_w=y_w1;
      trace_OUT=[trace_OUT;x,y];
   elseif WAY<4 & WAY==2
      x_w=x_w2;
      y_w=y_w2;
      trace_OUT=[trace_OUT;x,y];
   elseif WAY<4 & WAY==3
      x_w=x_w3;
      y_w=y_w3;
      trace_OUT=[trace_OUT;x,y];
   elseif WAY>3
      if R>L
         heading=heading+turn;
      elseif R<L
         heading=heading-turn;  
      end 
      trace_HOME=[trace_HOME;x,y]; 
   end

   dx=x_w-x;
   dy=y_w-y;
   d=sqrt(dx*dx+dy*dy);
   if dx==0.0 & dy>=0.0
	theta=pi/2;
   elseif dx==0.0 & dy<=0.0
	theta=-pi/2;
   elseif dy==0.0 & dx>=0.0
	theta=0.0;
   elseif dy==0.0 & dx<=0.0
	theta=pi;
   elseif dx<0.0 & dy>0.0
	theta=atan(dy/dx)+pi;
   elseif dx<0.0 & dy<0.0
	theta=atan(dy/dx)+pi;
   elseif dx>0.0 & dy<0.0
	theta=atan(dy/dx)+2*pi;
   else
	theta=atan(dy/dx);
   end
   heading_w=theta*180/pi;
   while heading_w>360
	heading_w=heading_w-360;
   end
   while heading_w<0
	heading_w=heading_w+360;
   end
   w=heading_w-heading;
   while w>360
      w=w-360;
   end
   while w<0
      w=w+360;
   end
   if WAY<4
      if w>180 & w<360
         heading=heading-turn;
      elseif w>0 & w<180
         heading=heading+turn;
      end
   end

   if d<DISTANCE & WAY<4
      WAY=WAY+1;
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
plot(trace_OUT(:,1),trace_OUT(:,2));
hold on;
plot(trace_HOME(:,1),trace_HOME(:,2),'g');
hold on;
plot(x_w1,y_w1,'r+');
hold on;
plot(x_w2,y_w2,'r+');
hold on;
plot(x_w3,y_w3,'r+');

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
