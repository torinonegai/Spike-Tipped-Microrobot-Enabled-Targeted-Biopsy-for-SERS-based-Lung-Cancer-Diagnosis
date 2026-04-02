function correct_signal = cosmic_ray_removal(data)
% 宇宙射线去除
% input signal [batch*wave_numbers]
[numbers, shift_length] = size(data);
correct_signal = data;
for i  = 1:numbers
    signal = data(i,:);
    [pks,locs,w,p] = findpeaks(signal);
    [~,I] = sort(p,"descend");
    I = I(1:max(5,length(I)));
    peak_matrix = [pks(I);locs(I);w(I);p(I);p(I)./w(I);];
    peak_matrix = peak_matrix(2:3, peak_matrix(5,:)>180 & peak_matrix(3,:)<6); % 峰高宽比阈值, 半峰宽<2.5
    loc = peak_matrix(1,:);hw = peak_matrix(2,:);
    for j = 1:size(peak_matrix,2)
        Endpoint = [max(1,round(loc(j)-2.5*hw(j))),min(shift_length,round(loc(j)+2.5*hw(j)))];
        segment = interp1(Endpoint,signal(Endpoint),Endpoint(1):Endpoint(2)); % 线性插值
        signal(Endpoint(1):Endpoint(2)) = segment;
    end
    correct_signal(i,:) = signal;
end
end