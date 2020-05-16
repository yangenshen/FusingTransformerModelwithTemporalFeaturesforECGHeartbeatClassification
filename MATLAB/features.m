DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230];
DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234];

lines = [];
for i = 1:22
    fprintf(strcat('--',num2str(i),'--\n'));
    seg_file = strcat(num2str(DS1(i)),'_seg.txt');
    data = csvread(seg_file);
    for j = 1:size(data,1)
       line = [data(j,1:3) data(j,4:end)];       
       lines = [lines;line];
    end 
end

train_file = 'train_features_all.txt';
csvwrite(train_file,lines);

lines = [];
for i = 1:22
    fprintf(strcat('--',num2str(i),'--\n'));
    seg_file = strcat(num2str(DS2(i)),'_seg.txt');
    data = csvread(seg_file);
    for j = 1:size(data,1)
        line = [data(j,1:3) data(j,4:end)];
        lines = [lines;line];
    end
end
test_file = 'test_features_all.txt';
csvwrite(test_file,lines);