function EVAL=Evaluate(Q,Testdata,Testlabel,svdd,numofmodes)
    predict_label = cell(1,numofmodes);
    for i = 1:numofmodes
        RedTestdata=Q{i} * Testdata{i};
        results = svdd.test(RedTestdata', Testlabel);
        predict_label{i} = results.predictedLabel;
    end

    EVAL = [];
    %and
    Decission_and = zeros(size(predict_label{1},1),1);
    for i = 1:numofmodes
        Decission_and = Decission_and + predict_label{i};
    end
    Decission_and(Decission_and ~= numofmodes) = -1;
    Decission_and(Decission_and == numofmodes) = 1;
    mic_and= Evaluate_mic(Testlabel,Decission_and);
    EVAL = cat(2,EVAL,mic_and);
    
    %or
    Decission_or = zeros(size(predict_label{1},1),1);
    for i = 1:numofmodes
        Decission_or = Decission_or + predict_label{i};
    end
    Decission_or(Decission_or ~= -numofmodes) = 1;
    Decission_or(Decission_or == -numofmodes) = -1;
    mic_or= Evaluate_mic(Testlabel,Decission_or);
    EVAL = cat(2,EVAL,mic_or);
    
    %m
    for i = 1:numofmodes
        mic= Evaluate_mic(Testlabel,predict_label{i});
        EVAL = cat(2,EVAL,mic);
    end
end
function EV = Evaluate_mic(ACTUAL,PREDICTED)
    idx = (ACTUAL()==1);
    p = length(ACTUAL(idx));
    n = length(ACTUAL(~idx));
    N = p+n;
    tp = sum(ACTUAL(idx)==PREDICTED(idx));
    tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
    fp = n-tn;
    fn = p-tp;
    tp_rate = tp/p;
    tn_rate = tn/n;
    accuracy = (tp+tn)/N;
    sensitivity = tp_rate;
    specificity = tn_rate;
    precision = tp/(tp+fp);
    precision(isnan(precision))=0;
    recall = sensitivity;
    f_measure = 2*((precision*recall)/(precision + recall));
    f_measure(isnan(f_measure))=0;
    sensitivity(isnan(sensitivity))=0; 
    tp_rate(isnan(tp_rate))=0;
    tn_rate(isnan(tn_rate))=0;
    gmean = sqrt(tp_rate*tn_rate);
    avg_senspec=((sensitivity+specificity)/2);
    EV = [accuracy sensitivity specificity precision recall f_measure avg_senspec gmean];
end
