function [unwph,iter,erglist] = puma_ho(psi,p)


%   Default values
potential.quantized     = 'yes';
potential.threshold     = 0;
cliques                 = [1 0;0 1];
qualitymaps             = repmat(zeros(size(psi,1),size(psi,2)),[1,1,2]); qual=0;
schedule                = 1;


if (qual==1)&&(size(qualitymaps,3)~=size(cliques,1))
    error('qualitymaps must be a 3D matrix whos 3D size is equal to no. cliques. Each plane on qualitymaps corresponds to a clique.');
end

% INPUT AND INITIALIZING %%%%%%%%%%%%%%%%%%%%%%%%
th    = getfield(potential,'threshold');
quant = getfield(potential,'quantized');


[m,n] = size(psi);
kappa = zeros(m,n); 
kappa_aux = kappa;
iter = 0;
erglist = [];

[cliquesm,~] = size(cliques); % Size of input cliques
if qual ==0
    qualitymaps             = repmat(zeros(size(psi,1),size(psi,2)),[1,1,size(cliques,1)]);
end
disc_bar = 1 - qualitymaps;
maxdesl = max(max(abs(cliques)));
base = zeros(2*maxdesl+2+m,2*maxdesl+2+n); base(maxdesl+2:maxdesl+2+m-1,maxdesl+2:maxdesl+2+n-1) = ones(m,n);

% PROCESSING   %%%%%%%%%%%%%%%%%%%%%%%%
for jump_size = schedule

    possible_improvment = 1;
    erg_previous = energy_ho(kappa,psi,base,p,cliques,disc_bar,th,quant);

    while possible_improvment
        iter = iter + 1;
        erglist = [erglist erg_previous];
        remain = [];
        base_kappa = zeros(2*maxdesl+2+m,2*maxdesl+2+n); base_kappa(maxdesl+2:maxdesl+2+m-1,maxdesl+2:maxdesl+2+n-1) = kappa;
        psi_base = zeros(2*maxdesl+2+m,2*maxdesl+2+n); psi_base(maxdesl+2:maxdesl+2+m-1,maxdesl+2:maxdesl+2+n-1) = psi;

        for t = 1:cliquesm
            base_start(:,:,t) = circshift(base,[-cliques(t,1),-cliques(t,2)]).*base;
            base_end(:,:,t) = circshift(base,[cliques(t,1),cliques(t,2)]).*base;

            auxili = circshift(base_kappa,[cliques(t,1),cliques(t,2)]);
            t_dkappa(:,:,t) = (base_kappa-auxili);
            auxili2 = circshift(psi_base,[cliques(t,1),cliques(t,2)]);
            dpsi = auxili2 - psi_base;
            
            a(:,:,t) = (2*pi*t_dkappa(:,:,t)-dpsi).*base.*circshift(base,[cliques(t,1),cliques(t,2)]);
            A(:,:,t) = clique_energy_ho(abs(a(:,:,t)),p,th,quant).*base.*circshift(base,[cliques(t,1),cliques(t,2)]);
            D(:,:,t) = A(:,:,t);
            C(:,:,t) = clique_energy_ho(abs(2*pi*jump_size + a(:,:,t)),p,th,quant).*base.*circshift(base,[cliques(t,1),cliques(t,2)]);
            B(:,:,t) = clique_energy_ho(abs(-2*pi*jump_size + a(:,:,t)),p,th,quant).*base.*circshift(base,[cliques(t,1),cliques(t,2)]);

            source(:,:,t) = circshift((C(:,:,t)-A(:,:,t)).*((C(:,:,t)-A(:,:,t))>0),[-cliques(t,1),-cliques(t,2)]).*base_start(:,:,t);
            sink(:,:,t) = circshift((A(:,:,t)-C(:,:,t)).*((A(:,:,t)-C(:,:,t))>0),[-cliques(t,1),-cliques(t,2)]).*base_start(:,:,t);

            source(:,:,t) = source(:,:,t) + ((D(:,:,t)-C(:,:,t)).*((D(:,:,t)-C(:,:,t))>0)).*base_end(:,:,t);
            sink(:,:,t) = sink(:,:,t) + ((C(:,:,t)-D(:,:,t)).*((C(:,:,t)-D(:,:,t))>0)).*base_end(:,:,t);
        end


        % We get rid of the "pass-partous"
        source(1:maxdesl+1,:,:)=[]; source(m+1:m+maxdesl+1,:,:)=[]; source(:,1:maxdesl+1,:)=[]; source(:,n+1:n+maxdesl+1,:)=[];
        sink(1:maxdesl+1,:,:)=[]; sink(m+1:m+maxdesl+1,:,:)=[];sink(:,1:maxdesl+1,:)=[]; sink(:,n+1:n+maxdesl+1,:)=[];
        auxiliar1 = B + C - A - D;
        auxiliar1(1:maxdesl+1,:,:)=[]; auxiliar1(m+1:m+maxdesl+1,:,:)=[]; auxiliar1(:,1:maxdesl+1,:)=[]; auxiliar1(:,n+1:n+maxdesl+1,:)=[];
        base_start(1:maxdesl+1,:,:)=[]; base_start(m+1:m+maxdesl+1,:,:)=[]; base_start(:,1:maxdesl+1,:)=[]; base_start(:,n+1:n+maxdesl+1,:)=[];
        base_end(1:maxdesl+1,:,:)=[]; base_end(m+1:m+maxdesl+1,:,:)=[]; base_end(:,1:maxdesl+1,:)=[]; base_end(:,n+1:n+maxdesl+1,:)=[];

        % We construct the "remain" and the "sourcesink" matrices
        for t=1:cliquesm
            start = find(base_start(:,:,t)~=0); endd = find(base_end(:,:,t)~=0);
            auxiliar2 = auxiliar1(:,:,t);
            auxiliar3 = [start endd  auxiliar2(endd) zeros(size(endd,1),1)];
            remain = [remain; auxiliar3];
        end
        %remain = sortrows(remain,[1 2]);
        sourcefinal = sum(source,3);
        sinkfinal = sum(sink,3);
        sourcesink = [(1:m*n)' sourcefinal(:) sinkfinal(:)];

        % KAPPA RELABELING
        [~,cutside] = mincut(sourcesink,remain);
        kappa_aux(cutside(:,1)) = kappa(cutside(:,1)) + (1 - cutside(:,2))*jump_size;

        % CHECK ENERGY IMPROVEMENT
        erg_actual = energy_ho(kappa_aux,psi,base,p,cliques,disc_bar,th,quant);
        %
        if (erg_actual < erg_previous)
            erg_previous = erg_actual;
            kappa = kappa_aux;
        else
            possible_improvment = 0;
            unwph = 2*pi*kappa + psi;
        end
        clear base_start base_end source sink auxiliar1 auxiliar2 A B C D;
    end
end

end


% ========================== REFERENCES ===================================
%   For reference see:
%   J. Bioucas-Dias and G. Valadão, "Phase Unwrapping via Graph Cuts"
%   IEEE Transactions Image Processing, 2007.
%   The algorithm here coded corresponds to a generalization for any
%   cliques set (not only vertical and horizontal).
%
%   J. Bioucas-Dias and J. Leitão, "The ZpiM Algorithm for Interferometric Image Reconstruction
%   in SAR/SAS", IEEE Transactions Image Processing, vol. 20, no. Y, 2001.
%
%   The technique here employed is also based on the article:
%   V. kolmogorov and R. Zabih, "What Energy Functions can be Minimized via Graph Cuts?",
%   European Conference on Computer Vision, May 2002.