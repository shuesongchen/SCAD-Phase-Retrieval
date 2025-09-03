function [xs] = Shrink_SCAD(x, gamma, lambda, eta)
    s = abs(x);
    
    if gamma > 1 + eta
        M1 = s >= gamma * lambda;
        M2 = s < gamma * lambda & s >= (1 + eta) * lambda;
        M3 = s < (1 + eta) * lambda & s >= eta * lambda;
        M0 = s < eta * lambda;
        ss = ((gamma - 1) * x - eta * gamma * lambda .* exp(1i*angle(x))) / (gamma - 1 - eta);
        xs = x .* M1 + ss .* M2 + (x - eta * lambda .* exp(1i*angle(x))) .* M3;
        xs(M0) = 0;
    else
        M1 = s > gamma * lambda;
        M2 = s <= gamma * lambda & s > eta * lambda;
        M0 = s <= eta * lambda;
        xs = x .* M1 + (x - eta * lambda .* exp(1i*angle(x))) .* M2;
        xs(M0) = 0;
    end
end