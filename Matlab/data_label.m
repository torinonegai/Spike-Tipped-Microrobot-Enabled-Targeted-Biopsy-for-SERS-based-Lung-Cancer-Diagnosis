clear;
name_cell = {"Ecoli","Bmega","Bthu","Sepid"}; % data name should correspond to fillname
num_class = length(name_cell);
option = "test";
%% Load data
data = []; label = [];
datapath = "Raman Data/"+option;
UUCLabel = [];


%%
% for UUCLabel = 0:3
data = []; label = [];
parfor i = 1:num_class % 四种细菌
    if i == UUCLabel+1
        continue
    end
    namelist = dir(datapath+"/"+name_cell{i}+"*"); % 文件名以该细菌开头的文件
    for j = 1:length(namelist) % 遍历名单中的文件
        % Load data
        tmp = readmatrix(datapath+"/"+namelist(j).name); % Read data
        tmp = cosmic_ray_removal(tmp);
        tmp = arPLS_baseline_correction(tmp,1e+7,1e-3);
        tmp = smoothdata(tmp,2,"sgolay",11);
        % tmp = normalize(tmp,2,"range",[0,1]);
        % tmp = normalize(tmp,2,"norm",2);
        tmp = normalize(tmp,2);
        % close all
        % plot(1:792,tmp(2:3:end,:));
        num_data = size(tmp,1)/3;
        data = [data;tmp];
        label = [label;(i-1)*ones(num_data,1)];
    end
end

%% save to txt files
if ~isempty(UUCLabel)
    save(option+"-"+num2str(UUCLabel)+"_data.txt","data","-ascii")
    fileID = fopen(option+"-"+num2str(UUCLabel)+"_label.txt",'w');
    fprintf(fileID,'%d\r\n',label);
    fclose(fileID);
else
    save(option+"_data.txt","data","-ascii")
    fileID = fopen(option+"_label.txt",'w');
    fprintf(fileID,'%d\r\n',label);
    fclose(fileID);
end
% end
