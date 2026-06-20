! ============================================================================
!  bfm_explicit_f.f90 -- Fortran ports of the explicit-solver hot kernels
!  (pressure Jacobi, momentum predictor, k-w SST).  Bit-faithful transcriptions
!  of the Numba kernels in solver/core_bfm.py, same operation order, no fastmath.
!  Built with f2py (see solver/simple_solver-style build); core_bfm tries this
!  module first, then falls back to the Numba kernels, then NumPy.
!
!  Arrays are (n_eta, n_xi) Fortran-order; j=radial (1=wall, n_eta=outer),
!  i=circumferential (periodic).
! ============================================================================

! ---- pressure-Poisson Jacobi (periodic E/W, Dirichlet outer p=0, Neumann wall)
subroutine jacobi_press(press, cE, cW, cN, cS, coeff, rhs, omega, nsweep, neta, nxi)
  implicit none
  integer, intent(in) :: neta, nxi, nsweep
  double precision, intent(in) :: omega
  double precision, intent(inout) :: press(neta, nxi)
  double precision, intent(in) :: cE(neta,nxi), cW(neta,nxi), cN(neta,nxi)
  double precision, intent(in) :: cS(neta,nxi), coeff(neta,nxi), rhs(neta,nxi)
!f2py intent(in,out) :: press
  double precision :: new(neta, nxi), pN, pS, pn_, s, m
  integer :: sw, j, i, ip, im
  do sw = 1, nsweep
    do i = 1, nxi
      ip = i + 1; if (ip > nxi) ip = 1
      im = i - 1; if (im < 1)   im = nxi
      do j = 1, neta
        if (j + 1 <= neta) then
          pN = press(j + 1, i)
        else
          pN = 0.0d0
        end if
        if (j - 1 >= 1) then
          pS = press(j - 1, i)
        else
          pS = press(j, i)
        end if
        pn_ = (cE(j,i)*press(j,ip) + cW(j,i)*press(j,im) &
             + cN(j,i)*pN + cS(j,i)*pS - rhs(j,i)) / coeff(j,i)
        if (pn_ > 5.0d0) then
          pn_ = 5.0d0
        else if (pn_ < -5.0d0) then
          pn_ = -5.0d0
        end if
        new(j, i) = (1.0d0 - omega)*press(j, i) + omega*pn_
      end do
    end do
    s = 0.0d0
    do i = 1, nxi
      do j = 1, neta
        s = s + new(j, i)
      end do
    end do
    m = s / dble(neta*nxi)
    do i = 1, nxi
      do j = 1, neta
        press(j, i) = new(j, i) - m
      end do
    end do
  end do
end subroutine jacobi_press

! ---- momentum predictor: central/central_jst advection + point-implicit diffusion
subroutine mom_predict(u, v, nu_t, nxE, nyE, dsE, dnE, nxW, nyW, dsW, dnW, &
                       nxN, nyN, dsN, dnN, nxS, nyS, dsS, dnS, &
                       cell_area, dE_circ, u_out, v_out, dt_arr, nu, &
                       jst_eps4, jst_on, u_star, v_star, neta, nxi)
  implicit none
  integer, intent(in) :: neta, nxi, jst_on
  double precision, intent(in) :: nu, jst_eps4
  double precision, intent(in) :: u(neta,nxi), v(neta,nxi), nu_t(neta,nxi)
  double precision, intent(in) :: nxE(neta,nxi), nyE(neta,nxi), dsE(neta,nxi), dnE(neta,nxi)
  double precision, intent(in) :: nxW(neta,nxi), nyW(neta,nxi), dsW(neta,nxi), dnW(neta,nxi)
  double precision, intent(in) :: nxN(neta,nxi), nyN(neta,nxi), dsN(neta,nxi), dnN(neta,nxi)
  double precision, intent(in) :: nxS(neta,nxi), nyS(neta,nxi), dsS(neta,nxi), dnS(neta,nxi)
  double precision, intent(in) :: cell_area(neta,nxi), dE_circ(neta,nxi)
  double precision, intent(in) :: u_out(nxi), v_out(nxi), dt_arr(neta,nxi)
  double precision, intent(out) :: u_star(neta,nxi), v_star(neta,nxi)
  integer :: j, i, ip, im, ip2, im2
  double precision :: uc, vc, uE, vE, uW, vW, uN, vN, uS, vS
  double precision :: unE, unW, unN, unS, uEf, vEf, uWf, vWf, uNf, vNf, uSf, vSf
  double precision :: ca, adv_u, adv_v, spd, d4u, d4v
  double precision :: ddE, ddW, ddN, ddS, nbrs_u, nbrs_v, eff, sum_ds_dn, dtc, denom
  do j = 1, neta
    do i = 1, nxi
      ip = i + 1; if (ip > nxi) ip = 1
      im = i - 1; if (im < 1)   im = nxi
      ip2 = i + 2; if (ip2 > nxi) ip2 = ip2 - nxi
      im2 = i - 2; if (im2 < 1)  im2 = im2 + nxi
      uc = u(j,i); vc = v(j,i)
      uE = u(j,ip); vE = v(j,ip); uW = u(j,im); vW = v(j,im)
      if (j + 1 <= neta) then
        uN = u(j+1,i); vN = v(j+1,i)
      else
        uN = u_out(i); vN = v_out(i)
      end if
      if (j - 1 >= 1) then
        uS = u(j-1,i); vS = v(j-1,i)
      else
        uS = -uc; vS = -vc
      end if
      unE = 0.5d0*(uc+uE)*nxE(j,i) + 0.5d0*(vc+vE)*nyE(j,i)
      unW = 0.5d0*(uc+uW)*nxW(j,i) + 0.5d0*(vc+vW)*nyW(j,i)
      unN = 0.5d0*(uc+uN)*nxN(j,i) + 0.5d0*(vc+vN)*nyN(j,i)
      unS = 0.5d0*(uc+uS)*nxS(j,i) + 0.5d0*(vc+vS)*nyS(j,i)
      uEf = 0.5d0*(uc+uE); vEf = 0.5d0*(vc+vE)
      uWf = 0.5d0*(uc+uW); vWf = 0.5d0*(vc+vW)
      uNf = 0.5d0*(uc+uN); vNf = 0.5d0*(vc+vN)
      uSf = 0.5d0*(uc+uS); vSf = 0.5d0*(vc+vS)
      ca = cell_area(j,i)
      adv_u = (unE*uEf*dsE(j,i) + unW*uWf*dsW(j,i) + unN*uNf*dsN(j,i) + unS*uSf*dsS(j,i))/ca
      adv_v = (unE*vEf*dsE(j,i) + unW*vWf*dsW(j,i) + unN*vNf*dsN(j,i) + unS*vSf*dsS(j,i))/ca
      if (jst_on /= 0) then
        spd = sqrt(uc*uc + vc*vc) + 1.0d-12
        d4u = u(j,ip2) - 4.0d0*u(j,ip) + 6.0d0*uc - 4.0d0*u(j,im) + u(j,im2)
        d4v = v(j,ip2) - 4.0d0*v(j,ip) + 6.0d0*vc - 4.0d0*v(j,im) + v(j,im2)
        adv_u = adv_u + jst_eps4*spd*d4u/(dE_circ(j,i) + 1.0d-12)
        adv_v = adv_v + jst_eps4*spd*d4v/(dE_circ(j,i) + 1.0d-12)
      end if
      ddE = dsE(j,i)/dnE(j,i); ddW = dsW(j,i)/dnW(j,i)
      ddN = dsN(j,i)/dnN(j,i); ddS = dsS(j,i)/dnS(j,i)
      nbrs_u = uE*ddE + uW*ddW + uN*ddN + uS*ddS
      nbrs_v = vE*ddE + vW*ddW + vN*ddN + vS*ddS
      eff = (nu + nu_t(j,i))/ca
      sum_ds_dn = ddE + ddW + ddN + ddS
      dtc = dt_arr(j,i)
      denom = 1.0d0 + dtc*eff*sum_ds_dn + 1.0d-30
      u_star(j,i) = (uc - dtc*adv_u + dtc*eff*nbrs_u)/denom
      v_star(j,i) = (vc - dtc*adv_v + dtc*eff*nbrs_v)/denom
    end do
  end do
end subroutine mom_predict
