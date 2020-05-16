DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230];
DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234];

%DS1
for i = 1:22
    data_file = strcat('data/', num2str(DS1(i)),'_data.mat');
    data = load(data_file);
    data = data.M(:,1); 
    data = median_filter(data); 
    data=low_pass(data);
%     data = dwt_data(data);
    save(strcat('denoised_',num2str(DS1(i)),'_data.mat'),'data');
end

% DS2
for i = 1:22
    data_file = strcat('data/', num2str(DS2(i)),'_data.mat');
    data = load(data_file);
    data = data.M(:,1); 
    data = median_filter(data); 
    data=low_pass(data);
%     data = dwt_data(data);
    save(strcat('denoised_',num2str(DS1(i)),'_data.mat'),'data');
end


%two-step median filter
function med_data = median_filter(data)
    fs = 250; % 360HZ
    tmp = data - medfilt1(data,fs*0.2);
    med_data = tmp - medfilt1(tmp,fs*0.6);
end

%12-order, 35-Hz cut-off, Attention: this func will casuse singal phase shift
function low_data = low_pass(data)
    fs = 250; % 360HZ
    order = 12;
    cut_off = 35;
    [b,a] = butter(order,cut_off/(fs/2),'low');
    low_data=filter(b,a,data);
end

function res = dwt_data(data)
    [c,l] = wavedec(data,7,'db6');
    ca11=appcoef(c,l,'db6',7); 
    cd1=detcoef(c,l,1);
    cd2=detcoef(c,l,2); 
    cd3=detcoef(c,l,3);
    cd4=detcoef(c,l,4);
    cd5=detcoef(c,l,5);
    cd6=detcoef(c,l,6);
    cd7=detcoef(c,l,7);
    sd1=zeros(1,length(cd1));
    sd2=zeros(1,length(cd2));
    sd3=zeros(1,length(cd3));
    sd4=wthresh(cd4,'s',0.014);
    sd5=wthresh(cd5,'s',0.014);
    sd6=wthresh(cd6,'s',0.014);
    sd7=wthresh(cd7,'s',0.014);
    c2=[ca11;sd7;sd6;sd5;sd4;sd3';sd2';sd1'];
    res=waverec(c2,l,'db6');
end


