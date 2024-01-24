function EVAL = DEMS_SVDD(obj)  
    warning("off");    
    if obj.Kernel_type == "gaussian"    
        for m = 1:obj.numofmodes
            [obj.Traindata{m}, obj.Testdata{m}] = kernel_rbf(obj.Traindata{m}, obj.Testdata{m}, obj.sigma);
        end
    end
    if obj.Kernel_type == "linear"
        for i=1:obj.numofmodes        
            tempQ = pca(obj.Traindata{i}');
            obj.Q{i}=tempQ(:,1:obj.d)';
            subspaceTraindata{i} = obj.Q{i} * obj.Traindata{i};
        end
    elseif obj.Kernel_type == "gaussian"
        kernel = Kernel('type', 'gauss', 'width', obj.sigma);     
        parameter = struct('application', 'dr', 'dim', obj.d,'display','off' ,'kernel', kernel);
        kpca = KernelPCA(parameter);
        for i=1:obj.numofmodes        
           tempQ = kpca.train(obj.Traindata{i}');
           obj.Q{i}=tempQ(:,1:obj.d)';
           subspaceTraindata{i} = obj.Q{i} * obj.Traindata{i};
        end
    end
    for iter = 1:obj.maxIter
        subspaceAlldata = [];subspaceAllLabel=[];
        for i=1:obj.numofmodes  
            subspaceAlldata = cat(2,subspaceAlldata,subspaceTraindata{i});
            subspaceAllLabel = cat(1,subspaceAllLabel,obj.Trainlabel);
        end
        W = GetSubspaceWeights(subspaceAlldata,subspaceAllLabel);
        if obj.Kernel_type=="linear"
            kernel = BaseKernel('type', "linear");
            svddParameter = struct('cost', obj.C,'kernelFunc', kernel,'weight',W,'display','off');
        elseif obj.Kernel_type=="gaussian"
            gamm = 1/(2*(obj.sigma^2)); 
            kernel = BaseKernel('type', 'gaussian','gamma',gamm);
            svddParameter = struct('cost', obj.C,'kernelFunc', kernel,'weight',W,'display','off');
        end
        obj.svdd = BaseSVDD(svddParameter);
        obj.svdd.train(subspaceAlldata',subspaceAllLabel);
        obj.AllAlphavector = obj.svdd.alpha;
        j=0;
        for i = 1:obj.numofmodes
            obj.Alphavector{i}=obj.AllAlphavector(j+1:j + size(obj.Trainlabel,1));
            j = j + size(obj.Trainlabel,2);
        end
        
        obj.Q = Update_Q(obj);
        for i = 1:obj.numofmodes
            [tmpQ, ~]=qr(obj.Q{i}',0);
            obj.Q{i} = tmpQ';
            tmpNorm = sqrt(diag(obj.Q{i} * obj.Q{i}'));
            obj.Q{i} = obj.Q{i}./(repmat(tmpNorm',size(obj.Q{i},2),1)');
            subspaceTraindata{i} = obj.Q{i} * obj.Traindata{i};
        end
    end
    subspaceAlldata = [];subspaceAllLabel=[];
    for i=1:obj.numofmodes  
        subspaceAlldata = cat(2,subspaceAlldata,subspaceTraindata{i});
        subspaceAllLabel = cat(1,subspaceAllLabel,obj.Trainlabel);
    end
    W = GetSubspaceWeights(subspaceAlldata,subspaceAllLabel);
    if obj.Kernel_type=="linear"
        kernel = BaseKernel('type', "linear");
        svddParameter = struct('cost', obj.C,'kernelFunc', kernel,'weight',W,'display','off');
    elseif obj.Kernel_type=="gaussian"
        gamm = 1/(2*(obj.sigma^2)); 
        kernel = BaseKernel('type', 'gaussian','gamma',gamm);
        svddParameter = struct('cost', obj.C,'kernelFunc', kernel,'weight',W,'display','off');
    end
    obj.svdd = BaseSVDD(svddParameter);
    obj.svdd.train(subspaceAlldata',subspaceAllLabel);
    EVAL = Evaluate(obj.Q,obj.Testdata,obj.Testlabel,obj.svdd,obj.numofmodes);
    
end