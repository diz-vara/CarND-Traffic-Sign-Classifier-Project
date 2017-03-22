# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:12:14 2017

@author: avarfolomeev
"""



def splitIndicies (Index, percent,step=0):
    out0 = np.empty(0,np.int32);
    out1 = np.empty(0,np.int32);

    nClasses = len(Index);

    for _class in range (nClasses):
        split = int(len(Index[_class]) * percent / 100);
        if split < 1:
            split = 1;
        
        o0,o1,o2 = np.split(Index[_class], [split*step, split*(step+1)])
        out0 = np.append(out0,o1);
        out1 = np.append(out1,o0);
        out1 = np.append(out1,o2);

    return out0, out1
    

classIndicies = [np.where(Y == i)[0] for i in range(n_classes)];
idxDevTest,idxDev = splitIndicies(classIndicies,10);

np.random.shuffle(idxDev);

devIndicies = [idxDev[np.where(Y[idxDev] == i)[0]] for i in range(n_classes)]


idxVal,idxTrn = splitIndicies(devIndicies,10,step)
