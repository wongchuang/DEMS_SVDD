function W = GetSubspaceWeights(SubspaceData,subspaceAllLabel)
        N = size(SubspaceData,2);
        NormalSize = 0;
        for i = 1:N
            if subspaceAllLabel(i) == 1
                NormalSize = NormalSize + 1;
            end
        end
        tempii = zeros(N,1);
        tempij = zeros(N,1);
        tempjk = 0;
        for i = 1:N
            tempii(i) = SubspaceData(:,i)' * SubspaceData(:,i);
            for j = 1:N
                if subspaceAllLabel(j) == 1
                    tempij(i) = tempij(i) + SubspaceData(:,i)' * SubspaceData(:,j);
                end
            end 
            if subspaceAllLabel(i) == 1
                tempjk = tempjk + tempij(i);
            end
        end
        SumNormalDi = 0;
        SumAbnormalDi = 0;
        Di = zeros(N,1);
        for i = 1:N
            Di(i) = tempii(i) - (2*tempij(i))/ NormalSize + tempjk / (NormalSize*NormalSize);
            if subspaceAllLabel(i) == 1
                SumNormalDi = SumNormalDi +Di(i);
            end
            if subspaceAllLabel(i) == -1
                SumAbnormalDi = SumAbnormalDi +Di(i);
            end
        end
        sumTi = 0;
        W = zeros(N,1);
        for i = 1:N
            if subspaceAllLabel(i) == -1
                sumTi = sumTi + SumAbnormalDi / Di(i);
            end
        end
        for i = 1:N
            if subspaceAllLabel(i) == 1
                pi = Di(i)/SumNormalDi;
                
            end
            if subspaceAllLabel(i) == -1
                pi = (SumAbnormalDi / Di(i)) / sumTi;
            end
            
            W(i) = -pi * log2(pi);
        end
end