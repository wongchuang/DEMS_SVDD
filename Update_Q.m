function QQQ = Update_Q(obj)
    M_sample_num = size(obj.Traindata{1},2); 
    G_mp= zeros(M_sample_num,M_sample_num); 
    D_mp = zeros(M_sample_num,M_sample_num);
    for i = 1:M_sample_num
        for j =1:M_sample_num
            if j>i
                break
            end
            if obj.Trainlabel(i) == 1 && obj.Trainlabel(j) == 1
                G_mp(i,j) = 1; 
                G_mp(j,i) = 1; 
            end
        end
    end
    for i = 1:M_sample_num            
        D_mp(i, i) = sum(G_mp(i, :));     
    end
    L_mp = D_mp - G_mp;       
    L_mm = cell(1,obj.numofmodes);
    for m = 1:obj.numofmodes      
        G_mm = zeros(M_sample_num,M_sample_num);
        D_mm = zeros(M_sample_num, M_sample_num);
        temp_M = zeros(M_sample_num, M_sample_num);
        for i = 1:M_sample_num  
            for j = 1:M_sample_num
                temp_M(i, j) = sum((obj.Traindata{m}(:, i) - obj.Traindata{m}(:, j)).^2);
            end
        end
        [~, index_M] = sort(temp_M, 2);   
        for i = 1:M_sample_num                           
            for j = 1:M_sample_num                 
                G_mm(i, index_M(i, j)) = exp(-temp_M(i, index_M(i, j)) ./ (2 .* (obj.delta .^ 2)));
            end
        end
        for i = 1:M_sample_num    
            D_mm(i, i) = sum(G_mm(i, :));  
        end
        L_mm{m} = D_mm - obj.beta .* G_mm;          
    end
    for m = 1:obj.numofmodes
        dm = size(obj.Q{m},2);
        S = zeros(size(obj.Traindata{m},1), size(obj.Traindata{m},1)); 
        small_value = 0.00001;
        temp = sum(obj.Q{m}.^2, 1)';
        for num = 1:size(obj.Traindata{m},1)
            S(num, num) = 1./(2 .* sqrt(temp(num, 1) + small_value));
        end
        QS = obj.Q{m} * S;     
        QXLX = zeros(obj.d,dm);
        yyaaQxx = zeros(obj.d,dm);
        yaQxx = zeros(obj.d,dm);
        for p = 1:obj.numofmodes
            if p~=m
                QXLX = QXLX + obj.Q{p} * obj.Traindata{p} * L_mp * (obj.Traindata{m}');
            else
                QXLX = QXLX + obj.Q{m} * obj.Traindata{m} * L_mm{m} * (obj.Traindata{m}');
            end
            
            for i = 1:M_sample_num    
                for j = 1:M_sample_num 
                    if (obj.Alphavector{m}(i)>0) && (obj.Alphavector{p}(j)>0)   
                        temp = obj.Trainlabel(i) * obj.Trainlabel(j) * obj.Alphavector{m}(i) * obj.Alphavector{p}(j);
                        yyaaQxx = yyaaQxx + temp .* ( obj.Q{p} * obj.Traindata{p}(:,j) * (obj.Traindata{m}(:,i)'));
                    end
                end
            end
        end
        for i = 1:M_sample_num 
            if obj.Alphavector{m}(i)>0           
                yaQxx = yaQxx +  (obj.Trainlabel(i) * obj.Alphavector{m}(i)) .* (obj.Q{m} * obj.Traindata{m}(:,i) * (obj.Traindata{m}(:,i)'));
            end
        end
        obj.Q{m}=obj.Q{m} - (2 * obj.eta) .* (obj.miu1 * QXLX + obj.miu2 * QS + yaQxx - yyaaQxx);
    end
    QQQ = obj.Q;
end