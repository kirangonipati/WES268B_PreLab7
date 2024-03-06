%% Prelab 7 

%% Delay Spread Model

FFT_Size = 32;
N = 1000;

% Random Sequence 
Data = randi([0 1],2*N*FFT_Size,1);
I_Data = Data(1:2:end);
Q_Data = Data(2:2:end);
QPSK_Data = (2*I_Data-1)+j*(2*Q_Data-1);

% IFFT Modulation without CP
QPSK_Data_for_IFFT = reshape(QPSK_Data,FFT_Size,[]);
QPSK_Data_after_IFFT =sqrt(FFT_Size)*ifft(QPSK_Data_for_IFFT);
Tx_Data = reshape(QPSK_Data_after_IFFT,1,[]);

% ISI Channel
h_channel = [1 zeros(1,14) -1 zeros(1,FFT_Size-17)];
Channel_Data_Conv = conv(Tx_Data,h_channel);
Channel_Data = Channel_Data_Conv(1:length(Tx_Data));

% FFT Demodulation for ISI channel without CP
N0_Data = zeros(size(Channel_Data));
Rx_Data = N0_Data+Channel_Data;
Rx_Data_fft = reshape(Rx_Data,FFT_Size,[]);
Demod_Data = 1/sqrt(FFT_Size)*fft(Rx_Data_fft);
Received_Data = reshape(Demod_Data,1,[]);

% Power Spectrum Density of ISI Channel and AWGN channel
figure;
subplot(211);
L = length(Rx_Data); NFFT = 2^nextpow2(L);
Y = abs(fft(Rx_Data,NFFT)/NFFT);
f_d = 2*pi/NFFT:2*pi/NFFT:2*pi;

plot(f_d,db(2*Y));
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');
title('Received QPSK ISI Channel spectrum');
xlim([0 2*pi]);

subplot(212);
L = length(Channel_Data_Conv); NFFT = 2^nextpow2(L);
Y = abs(fft(Tx_Data,NFFT)/NFFT);
f_d = 2*pi/NFFT:2*pi/NFFT:2*pi;
plot(f_d,db(2*Y));
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');
title('Transmitted QPSK spectrum');
xlim([0 2*pi]);

% Without CP, constellation plot
sub1_Tx = QPSK_Data(1:FFT_Size:end); sub1_Rx = Received_Data(1:FFT_Size:end);
sub15_Tx = QPSK_Data(15:FFT_Size:end);sub15_Rx = Received_Data(15:FFT_Size:end);
sub26_Tx = QPSK_Data(26:FFT_Size:end);sub26_Rx = Received_Data(26:FFT_Size:end);

h = scatterplot(sub1_Rx,1,0,'g.');
hold on;
scatterplot(sub1_Tx,1,0,'k*',h);
hold off;
legend('Distorted Constellation','Orignial QPSK constellation');
xlim([-3 3]);
ylim([-3 +3]);
title({['Plot of 1st subcarrier constellation of Transmitted Data'];['without CP']});
h = scatterplot(sub15_Rx,1,0,'g.');
hold on;
scatterplot(sub15_Tx,1,0,'k*',h);
hold off;
legend('Distorted Constellation','Orignial QPSK constellation');
xlim([-3 3]);
ylim([-3 +3]);
title({['Plot of the 15th subcarrier constellation of Transmitted Data'];['without CP']});
h = scatterplot(sub26_Rx,1,0,'g.');
hold on;
scatterplot(sub26_Tx,1,0,'k*',h);
hold off;
title({['Plot of 26th subcarrier constellation of Transmitted Data'];['without CP']});
legend('Distorted Constellation','Orignial QPSK constellation');
xlim([-3 3]);
ylim([-3 +3]);

% With CP
L = 16;
CP_Data = [QPSK_Data_after_IFFT(end-L+1:end,:);QPSK_Data_after_IFFT];
CP_Tx_Data = reshape(CP_Data,1,[]);
CP_Channel_Data_Conv = conv(CP_Tx_Data,h_channel);
CP_Channel_Data = CP_Channel_Data_Conv(1:length(CP_Tx_Data));

% discard CP and Demod
CP_Rx_Data_withCP = reshape(CP_Channel_Data,FFT_Size+L,[]);
CP_Rx_Data = CP_Rx_Data_withCP(L+1:end,:);
CP_Demod_Data = 1/sqrt(FFT_Size)*fft(CP_Rx_Data);
CP_Received_Data = reshape(CP_Demod_Data,1,[]);

% With CP, constellation plot of subcarriers compared to AWGN Channel
CP_sub2_Tx = QPSK_Data(2:FFT_Size:end);
CP_sub2_Rx = CP_Received_Data(2:FFT_Size:end);

CP_sub15_Tx = QPSK_Data(15:FFT_Size:end);
CP_sub15_Rx = CP_Received_Data(15:FFT_Size:end);

CP_sub31_Tx = QPSK_Data(31:FFT_Size:end);
CP_sub31_Rx = CP_Received_Data(31:FFT_Size:end);

h = scatterplot(CP_Received_Data,1,0,'g.');
hold on;
scatterplot(QPSK_Data,1,0,'k*',h);
hold off;
title({['Plot of total constellation of Transmitted Data'];['with CP']});
legend('CP received constellation','orignial QPSK constellation');

h = scatterplot(CP_sub2_Rx,1,0,'g.');
hold on;
scatterplot(CP_sub2_Tx,1,0,'k*',h);
hold off;
title({['Plot of 2nd subcarrier constellation of Transmitted Data'];['with CP']});
legend('CP received constellation','orignial QPSK constellation');
xlim([-2.5 2.5]);
ylim([-2.5 +2.5]);
h = scatterplot(CP_sub15_Rx,1,0,'g.');
hold on;
scatterplot(CP_sub15_Tx,1,0,'k*',h);
hold off;
title({['Plot of 15th subcarrier constellation of Transmitted Data'];['with CP']});
legend('CP received constellation','orignial QPSK constellation');
xlim([-2.5 2.5]);
ylim([-2.5 +2.5]);
h = scatterplot(CP_sub31_Rx,1,0,'g.');
hold on;
scatterplot(CP_sub31_Tx,1,0,'k*',h);
hold off;
title({['Plot of 31st subcarrier constellation of Transmitted Data'];['with CP']});
legend('CP received constellation','orignial QPSK constellation');
xlim([-2.5 2.5]);
ylim([-2.5 +2.5]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Channel Estimation and Equilization

load 'ofdm_pkt.mat';

% OFDM BPSK Training Sequence
nfft = 64;
ofdm_mask = [0 sign(1:26) (27:37)*0 sign(38:63)];
ofdm_idx = [ 1:26 38:63 ] + 1;
nCars = sum(ofdm_mask);
nOFDM_Symbols = 10;

T = ofdm_mask;
T(ofdm_idx) = s;
t = ifft(T) * sqrt(nfft * (nfft/nCars) );

% Apply cross-correlation
r = xcorr(y,t);
rr = abs(xcorr(y,t)) > (0.75*nfft);
corr_idx = find(rr == 1);

nGI = 32;
nCP = 16;
nfft = 64;
nTr1 = 64;
nTr2 = 64;
nTR = nTr1+nTr2;

n1 = corr_idx(1);
n2 = corr_idx(2);

% Freq Offset using n1 and n2
tt = y( (1:128) + (n1-1) );
t1 = tt(1:64);
t2 = tt(65:end);

rr = t1 .* conj(t2);
f_est = (1/2/pi/64) * angle( sum(rr)/64 );
f_alt = mean((1/2/pi/64) * angle( rr ));
fco = (1/2/pi/64) * angle( rr ); 
rr2 = conj(t1) .* t2;
fco1 = (1/2/pi/64) * angle( rr2 ); 

% Channel Estimation
T1 = (1/sqrt(nfft))*fft(t1 .* exp(1i*2*pi*f_est*(0:63))) * sqrt(nCars/nfft);
T2 = (1/sqrt(nfft))*fft(t2 .* exp(1i*2*pi*f_est*(64:127))) * sqrt(nCars/nfft);

% freq domain channel est averaging
H = T .* (T1+T2)/2;

% Frequency offset to entire packet
L = length(y(n1:end));
z = y(n1:end) .* exp(1i*2*pi*f_est*(-(64)+(0:L-1)));

% Start of OFDM symbol
z = z(1+nTR+nGI:end);                              

z_block = reshape(z, nCP+nfft, nOFDM_Symbols);

% Remove CP
z_nocp = z_block(1+nCP : end, :);

% demodulate using the fft
Z = (1/sqrt(nfft))*fft(z_nocp);

% Freq Domain Equalizer
Heq = H;
Heq(Heq==0)=1;
G = diag( 1 ./ Heq );

Z = reshape( G*Z, 1, nfft*nOFDM_Symbols);

% Constellation Plot of all subchannels
scatterplot(Z);
