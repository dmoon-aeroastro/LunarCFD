! k-w SST update (two-pass; bit-faithful with _sst_jit in core_bfm.py).
! Numba j=0 wall -> Fortran j=1; numba j=n_eta-1 outer -> Fortran j=neta.
subroutine sst_update(u, v, tke, tom, d_w, &
    nxE, nyE, dsE, dnE, nxW, nyW, dsW, dnW, nxN, nyN, dsN, dnN, nxS, nyS, dsS, dnS, &
    cell_area, u_out, v_out, nu, k_fs, om_fs, kappa, beta_str, b1, b2, &
    sk1, sk2, sw1, sw2, g1, g2, a1, nu_t_mult, pk_limit, S_max, dt_arr, &
    tke_new, tom_new, nu_t, neta, nxi)
  implicit none
  integer, intent(in) :: neta, nxi
  double precision, intent(in) :: nu, k_fs, om_fs, kappa, beta_str, b1, b2
  double precision, intent(in) :: sk1, sk2, sw1, sw2, g1, g2, a1, nu_t_mult, pk_limit, S_max
  double precision, intent(in) :: u(neta,nxi), v(neta,nxi), tke(neta,nxi), tom(neta,nxi), d_w(neta,nxi)
  double precision, intent(in) :: nxE(neta,nxi),nyE(neta,nxi),dsE(neta,nxi),dnE(neta,nxi)
  double precision, intent(in) :: nxW(neta,nxi),nyW(neta,nxi),dsW(neta,nxi),dnW(neta,nxi)
  double precision, intent(in) :: nxN(neta,nxi),nyN(neta,nxi),dsN(neta,nxi),dnN(neta,nxi)
  double precision, intent(in) :: nxS(neta,nxi),nyS(neta,nxi),dsS(neta,nxi),dnS(neta,nxi)
  double precision, intent(in) :: cell_area(neta,nxi), u_out(nxi), v_out(nxi), dt_arr(neta,nxi)
  double precision, intent(out) :: tke_new(neta,nxi), tom_new(neta,nxi), nu_t(neta,nxi)
  double precision :: tk(neta,nxi), om_w(nxi), bstr4
  double precision :: d1, k1, om_log, om_vis2, ow
  double precision :: uc,vc,uE,vE,uW,vW,uN,vN,uS,vS, kc,omc,kEv,kWv,omEv,omWv,kNv,omNv,kSv,omSv
  double precision :: unE,unW,unN,unS, ca_sst, uEf,vEf,uWf,vWf,uNf,vNf,uSf,vSf
  double precision :: dudx,dudy,dvdx,dvdy, S2, S, k_s, om_s
  double precision :: dkdx,dkdy,domdx,domdy, gk_gom, cd_t, CD_kw, dw
  double precision :: r1a,r1b,r1c, mx, arg1, F1, r2a,r2b,r2, F2, sk,sw,bbl,gbl
  double precision :: den_nt, nt, Pk_cap, Pk, ddE,ddW,ddN,ddS, ca, dtc
  double precision :: kE_uw,kW_uw,kN_uw,kS_uw, adv_k, Dk, nk, sdn, kn
  double precision :: xd, omE_uw,omW_uw,omN_uw,omS_uw, adv_om, Dom, nom, Pom_cap, Pom, on_
  integer :: i, j, ip, im
  bstr4 = beta_str**0.25d0
  tk = tke
  do i = 1, nxi
    d1 = d_w(1, i)
    if (tk(2,i) > 1.0d-10) then; k1 = tk(2,i); else; k1 = 1.0d-10; end if
    om_log = sqrt(k1) / (kappa*d1*bstr4)
    om_vis2 = (60.0d0*nu/(b1*d1*d1))**2
    ow = sqrt(om_vis2 + om_log*om_log)
    if (ow < 1.0d5) then; om_w(i) = ow; else; om_w(i) = 1.0d5; end if
    if (om_log > sqrt(om_vis2)) then; tk(1,i) = tk(2,i); else; tk(1,i) = 0.0d0; end if
  end do
  do j = 1, neta
    do i = 1, nxi
      ip = i + 1; if (ip > nxi) ip = 1
      im = i - 1; if (im < 1) im = nxi
      uc = u(j,i); vc = v(j,i); uE = u(j,ip); vE = v(j,ip); uW = u(j,im); vW = v(j,im)
      if (j+1 <= neta) then; uN = u(j+1,i); vN = v(j+1,i); else; uN = u_out(i); vN = v_out(i); end if
      if (j-1 >= 1) then; uS = u(j-1,i); vS = v(j-1,i); else; uS = -uc; vS = -vc; end if
      kc = tk(j,i); omc = tom(j,i); kEv = tk(j,ip); kWv = tk(j,im); omEv = tom(j,ip); omWv = tom(j,im)
      if (j+1 <= neta) then; kNv = tk(j+1,i); omNv = tom(j+1,i); else; kNv = k_fs; omNv = om_fs; end if
      if (j-1 >= 1) then; kSv = tk(j-1,i); omSv = tom(j-1,i); else; kSv = 0.0d0; omSv = om_w(i); end if
      unE = 0.5d0*(uc+uE)*nxE(j,i) + 0.5d0*(vc+vE)*nyE(j,i)
      unW = 0.5d0*(uc+uW)*nxW(j,i) + 0.5d0*(vc+vW)*nyW(j,i)
      unN = 0.5d0*(uc+uN)*nxN(j,i) + 0.5d0*(vc+vN)*nyN(j,i)
      unS = 0.5d0*(uc+uS)*nxS(j,i) + 0.5d0*(vc+vS)*nyS(j,i)
      if (cell_area(j,i) > 1.0d-6) then; ca_sst = cell_area(j,i); else; ca_sst = 1.0d-6; end if
      uEf = 0.5d0*(uc+uE); vEf = 0.5d0*(vc+vE); uWf = 0.5d0*(uc+uW); vWf = 0.5d0*(vc+vW)
      uNf = 0.5d0*(uc+uN); vNf = 0.5d0*(vc+vN); uSf = 0.5d0*(uc+uS); vSf = 0.5d0*(vc+vS)
      dudx = (uEf*nxE(j,i)*dsE(j,i)+uWf*nxW(j,i)*dsW(j,i)+uNf*nxN(j,i)*dsN(j,i)+uSf*nxS(j,i)*dsS(j,i))/ca_sst
      dudy = (uEf*nyE(j,i)*dsE(j,i)+uWf*nyW(j,i)*dsW(j,i)+uNf*nyN(j,i)*dsN(j,i)+uSf*nyS(j,i)*dsS(j,i))/ca_sst
      dvdx = (vEf*nxE(j,i)*dsE(j,i)+vWf*nxW(j,i)*dsW(j,i)+vNf*nxN(j,i)*dsN(j,i)+vSf*nxS(j,i)*dsS(j,i))/ca_sst
      dvdy = (vEf*nyE(j,i)*dsE(j,i)+vWf*nyW(j,i)*dsW(j,i)+vNf*nyN(j,i)*dsN(j,i)+vSf*nyS(j,i)*dsS(j,i))/ca_sst
      S2 = 2.0d0*dudx*dudx + 2.0d0*dvdy*dvdy + (dudy+dvdx)**2
      if (S2 < 0.0d0) S2 = 0.0d0
      S = sqrt(S2)
      if (S2 > S_max*S_max) S2 = S_max*S_max
      if (S > S_max) S = S_max
      if (kc > 0.0d0) then; k_s = kc; else; k_s = 0.0d0; end if
      if (omc > 1.0d-10) then; om_s = omc; else; om_s = 1.0d-10; end if
      dkdx = (0.5d0*(kc+kEv)*nxE(j,i)*dsE(j,i)+0.5d0*(kc+kWv)*nxW(j,i)*dsW(j,i) &
            + 0.5d0*(kc+kNv)*nxN(j,i)*dsN(j,i)+0.5d0*(kc+kSv)*nxS(j,i)*dsS(j,i))/ca_sst
      dkdy = (0.5d0*(kc+kEv)*nyE(j,i)*dsE(j,i)+0.5d0*(kc+kWv)*nyW(j,i)*dsW(j,i) &
            + 0.5d0*(kc+kNv)*nyN(j,i)*dsN(j,i)+0.5d0*(kc+kSv)*nyS(j,i)*dsS(j,i))/ca_sst
      domdx = (0.5d0*(omc+omEv)*nxE(j,i)*dsE(j,i)+0.5d0*(omc+omWv)*nxW(j,i)*dsW(j,i) &
            + 0.5d0*(omc+omNv)*nxN(j,i)*dsN(j,i)+0.5d0*(omc+omSv)*nxS(j,i)*dsS(j,i))/ca_sst
      domdy = (0.5d0*(omc+omEv)*nyE(j,i)*dsE(j,i)+0.5d0*(omc+omWv)*nyW(j,i)*dsW(j,i) &
            + 0.5d0*(omc+omNv)*nyN(j,i)*dsN(j,i)+0.5d0*(omc+omSv)*nyS(j,i)*dsS(j,i))/ca_sst
      gk_gom = dkdx*domdx + dkdy*domdy
      if (gk_gom /= gk_gom) gk_gom = 0.0d0
      if (gk_gom > 1.0d20) then; gk_gom = 1.0d20; else if (gk_gom < -1.0d20) then; gk_gom = -1.0d20; end if
      cd_t = 2.0d0*sw2/om_s*gk_gom
      if (cd_t > 1.0d-10) then; CD_kw = cd_t; else; CD_kw = 1.0d-10; end if
      dw = d_w(j,i)
      r1a = sqrt(k_s)/(beta_str*om_s*dw+1.0d-30)
      r1b = 500.0d0*nu/(om_s*dw*dw+1.0d-30)
      r1c = 4.0d0*sw2*k_s/(CD_kw*dw*dw+1.0d-30)
      if (r1a > r1b) then; mx = r1a; else; mx = r1b; end if
      if (mx < r1c) then; arg1 = mx; else; arg1 = r1c; end if
      if (arg1 > 50.0d0) arg1 = 50.0d0
      F1 = tanh(arg1**4)
      r2a = 2.0d0*sqrt(k_s)/(beta_str*om_s*dw+1.0d-30)
      r2b = 500.0d0*nu/(om_s*dw*dw+1.0d-30)
      if (r2a > r2b) then; r2 = r2a; else; r2 = r2b; end if
      if (r2 > 50.0d0) r2 = 50.0d0
      F2 = tanh(r2**2)
      sk = F1*sk1+(1.0d0-F1)*sk2; sw = F1*sw1+(1.0d0-F1)*sw2
      bbl = F1*b1+(1.0d0-F1)*b2; gbl = F1*g1+(1.0d0-F1)*g2
      if (a1*om_s > S*F2) then; den_nt = a1*om_s; else; den_nt = S*F2; end if
      nt = a1*k_s/den_nt
      if (nt > 1.0d4*nu) nt = 1.0d4*nu
      nt = nt*nu_t_mult
      if (j == 1) nt = 0.0d0
      nu_t(j,i) = nt
      Pk_cap = pk_limit*beta_str*k_s*om_s
      if (nt*S2 < Pk_cap) then; Pk = nt*S2; else; Pk = Pk_cap; end if
      ddE = dsE(j,i)/dnE(j,i); ddW = dsW(j,i)/dnW(j,i); ddN = dsN(j,i)/dnN(j,i); ddS = dsS(j,i)/dnS(j,i)
      ca = cell_area(j,i); dtc = dt_arr(j,i)
      if (unE > 0.0d0) then; kE_uw = kc; else; kE_uw = kEv; end if
      if (unW > 0.0d0) then; kW_uw = kc; else; kW_uw = kWv; end if
      if (unN > 0.0d0) then; kN_uw = kc; else; kN_uw = kNv; end if
      if (unS > 0.0d0) then; kS_uw = kc; else; kS_uw = kSv; end if
      adv_k = (unE*kE_uw*dsE(j,i)+unW*kW_uw*dsW(j,i)+unN*kN_uw*dsN(j,i)+unS*kS_uw*dsS(j,i))/ca
      Dk = (nu+sk*nt)/ca; nk = kEv*ddE+kWv*ddW+kNv*ddN+kSv*ddS; sdn = ddE+ddW+ddN+ddS
      kn = (kc-dtc*adv_k+dtc*Dk*nk+dtc*Pk)/(1.0d0+dtc*Dk*sdn+dtc*beta_str*om_s+1.0d-30)
      if (kn < 0.0d0) kn = 0.0d0
      xd = 2.0d0*(1.0d0-F1)*sw2/om_s*gk_gom
      if (xd < 0.0d0) xd = 0.0d0
      if (unE > 0.0d0) then; omE_uw = omc; else; omE_uw = omEv; end if
      if (unW > 0.0d0) then; omW_uw = omc; else; omW_uw = omWv; end if
      if (unN > 0.0d0) then; omN_uw = omc; else; omN_uw = omNv; end if
      if (unS > 0.0d0) then; omS_uw = omc; else; omS_uw = omSv; end if
      adv_om = (unE*omE_uw*dsE(j,i)+unW*omW_uw*dsW(j,i)+unN*omN_uw*dsN(j,i)+unS*omS_uw*dsS(j,i))/ca
      Dom = (nu+sw*nt)/ca; nom = omEv*ddE+omWv*ddW+omNv*ddN+omSv*ddS
      Pom_cap = pk_limit*bbl*om_s*om_s
      if (gbl*S2 < Pom_cap) then; Pom = gbl*S2; else; Pom = Pom_cap; end if
      on_ = (omc-dtc*adv_om+dtc*Dom*nom+dtc*Pom+dtc*xd)/(1.0d0+dtc*Dom*sdn+dtc*bbl*om_s+dtc*xd/om_s+1.0d-30)
      if (on_ < 1.0d-10) on_ = 1.0d-10
      if (kn > 1.0d0) kn = 1.0d0
      if (on_ > 1.0d6) on_ = 1.0d6
      if (j == 1) then
        kn = 0.0d0; on_ = om_w(i)
      else if (j == neta) then
        kn = k_fs; on_ = om_fs
      end if
      tke_new(j,i) = kn; tom_new(j,i) = on_
    end do
  end do
end subroutine sst_update
