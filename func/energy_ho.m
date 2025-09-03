function erg = energy_ho(kappa,psi,base,p,cliques,disc_bar,th,quant)


[m,n] = size(psi);
[cliquesm,~] = size(cliques); % Size of input cliques
maxdesl = max(max(abs(cliques))); 

base_kappa    = zeros(2*maxdesl+2+m,2*maxdesl+2+n); base_kappa(maxdesl+2:maxdesl+2+m-1,maxdesl+2:maxdesl+2+n-1) = kappa;
psi_base      = zeros(2*maxdesl+2+m,2*maxdesl+2+n); psi_base(maxdesl+2:maxdesl+2+m-1,maxdesl+2:maxdesl+2+n-1) = psi;
z = size(disc_bar,3);
base_disc_bar  = repmat(zeros(2*maxdesl+2+m,2*maxdesl+2+n),[1 1 z]); base_disc_bar(maxdesl+2:maxdesl+2+m-1,maxdesl+2:maxdesl+2+n-1,:) = disc_bar;

for t = 1:cliquesm
    % The allowed start and end pixels of the "interpixel" directed edge
    base_start(:,:,t) = circshift(base,[-cliques(t,1),-cliques(t,2)]).*base;
    base_end(:,:,t) = circshift(base,[cliques(t,1),cliques(t,2)]).*base;
    
    auxili = circshift(base_kappa,[cliques(t,1),cliques(t,2)]);
    t_dkappa(:,:,t) = (base_kappa-auxili);
    auxili2 = circshift(psi_base,[cliques(t,1),cliques(t,2)]);
    dpsi = auxili2 - psi_base;
    a(:,:,t) = (2*pi*t_dkappa(:,:,t)-dpsi).*base.*circshift(base,[cliques(t,1),cliques(t,2)])...
               .*base_disc_bar(:,:,t);
end

erg = sum(sum(sum((clique_energy_ho(a,p,th,quant)))));


end


% ========================== REFERENCES ===================================
%   (see J. Bioucas-Dias and G. Valadão, "Phase Unwrapping via Graph Cuts"
%   submitted to IEEE Transactions Image Processing, October, 2005).
%   SITE: www.lx.it.pt/~bioucas/ 