file_index = [100,101,102,103,104,105,106,107,108,109,111,112,113,114,115,116,117,118,119,121,122,123,124,200,201,202,203,205,207,208,209,210,212,213,214,215,217,219,220,221,222,223,228,230,231,232,233,234];

classN = ['N','L','R','e','j'];
classS = ['A','a','J','S'];
classV = ['V','E'];
classF = ['F'];
classQ = ['f','Q'];
for i = 1:48
    anno = {};
    file_name =  strcat('mitdb/',num2str(file_index(i)));
    [ann,anntype]=rdann(file_name,'atr',[]);
    cnt = 0;
    for j = 1:length(ann)
        if(ismember(anntype(j),classN))
            cnt = cnt + 1;
            anno{cnt,1} = ann(j);
            anno{cnt,2} = 'N';
        elseif(ismember(anntype(j),classS))
            cnt = cnt + 1;
            anno{cnt,1} = ann(j);
            anno{cnt,2} = 'S';
        elseif(ismember(anntype(j),classV))
            cnt = cnt + 1;
            anno{cnt,1} = ann(j);
            anno{cnt,2} = 'V';
        elseif(ismember(anntype(j),classF))
            cnt = cnt + 1;
            anno{cnt,1} = ann(j);
            anno{cnt,2} = 'F';
        elseif(ismember(anntype(j),classQ))
            cnt = cnt + 1;
            anno{cnt,1} = ann(j);
            anno{cnt,2} = 'Q';
        end
    end
    anno_file = strcat(num2str(file_index(i)),'_anno.mat');
    save(anno_file,'anno');
end
