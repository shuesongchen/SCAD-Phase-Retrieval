function [flow,cutside] = mincut(sourcesink,remain)

sourcesinkg = single(sourcesink);
remaing = single(remain);

[flowg,cutsideg] = mf2(sourcesinkg,remaing);
flow = double(flowg); 
cutside = double(cutsideg);

end