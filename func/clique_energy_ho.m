function e=clique_energy_ho(d,p,th,quant)


switch quant
    case 'no'        % non quantized potential
        d=abs(d);
    case 'yes'       % quantized potential (2pi Quantization of phase difference)
        d=abs(round(d/2/pi)*2*pi);
end

if th~=0
    mask = (d<=th);
    e = th^(p-2)*d.^2.*mask + d.^p.*(1-mask);
else
    e = d.^p;
end

end

