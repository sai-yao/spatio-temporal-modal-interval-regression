function me = per_prox(z,data,p,w)

me=data;
e=data-z;

in_ep = logical((z<=data) .* (e > p*w));
in_en = logical((z>data) .*  (e < -(1-p)*w));

me(in_ep) = z(in_ep) + p*w(in_ep);
me(in_en) = z(in_en) - (1-p)*w(in_en);

end

