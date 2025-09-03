function re = RPhase(x,A)

amp_x = abs(x);
pha_x = puma_ho(angle(x),1);

pha_A = angle(A);
pha_x = pha_x - mean(pha_x(:)) + mean(pha_A(:));

re = amp_x.*exp(1i*pha_x);

end
