DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230];
DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234];
class = ['N','S','V','F','Q'];

shift = 13;
pre = 100-shift;
post = 140+shift;
% DS1
for i = 1:22
    lines = [];
    seg_file = strcat(num2str(DS1(i)),'_seg.txt');
    anno_file = strcat(num2str(DS1(i)),'_anno.mat');
    data_file = strcat('denoised_',num2str(DS1(i)),'_data.mat');
    anno = load(anno_file);     anno = anno.anno;
    data = load(data_file);     data = data.data; 
    for j = 2:size(anno,1)-1
        label =  find(class == anno{j,2});
        %anondan Q 
        if label == 5
           continue;
        end
        
        % too long RR
        pre_RR = anno{j,1}-anno{j-1,1};
        post_RR = anno{j+1,1}-anno{j,1};
        
        if pre_RR > 500 || post_RR > 500
           continue; 
        end
        
        peak = anno{j,1};
        seg = data(peak-pre:peak+post)';
        line = [label pre_RR post_RR seg];
        lines = [lines;line];
    end
    csvwrite(seg_file,lines);
end

%DS2
for i = 1:22
    lines = [];
    seg_file = strcat(num2str(DS2(i)),'_seg.txt');
    anno_file = strcat(num2str(DS2(i)),'_anno.mat');
    data_file = strcat('denoised_',num2str(DS2(i)),'_data.mat');
    anno = load(anno_file);     anno = anno.anno;
    data = load(data_file);     data = data.data; 
    for j = 2:size(anno,1)-1
        label =  find(class == anno{j,2});
        %anondan Q 
        if label == 5
           continue;
        end
        
        % too long RR
        pre_RR = anno{j,1}-anno{j-1,1};
        post_RR = anno{j+1,1}-anno{j,1};
        
        if pre_RR > 500 || post_RR > 500
           continue; 
        end
        
        peak = anno{j,1};
        seg = data(peak-pre:peak+post)';
        line = [label pre_RR post_RR  seg];
        lines = [lines;line];
    end
    csvwrite(seg_file,lines);
end
