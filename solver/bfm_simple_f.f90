! ============================================================================
!  bfm_simple_f.f90  --  Fortran hot kernels for the #1B implicit SIMPLE solver
!
!  Built into a Python extension with f2py:
!     python -m numpy.f2py -c bfm_simple_f.f90 -m bfm_simple_f
!  (requires a Fortran compiler, e.g. gfortran).  The Python side
!  (solver/simple_solver.py) imports this module when present and falls back to
!  an identical Numba implementation otherwise, so results match either way.
!
!  Index convention mirrors the NumPy solver: arrays are (n_eta, n_xi) with
!     j = 1 .. n_eta   (radial; j=1 wall-adjacent, j=n_eta outer Dirichlet ring)
!     i = 1 .. n_xi    (circumferential, PERIODIC)
!  Arrays are passed in Fortran order (np.asfortranarray on the Python side) so
!  the (j,i) access here is contiguous and copy-free.
! ============================================================================

! ----------------------------------------------------------------------------
!  jacobi5 : under-relaxed Jacobi sweeps of the 5-point system
!     ap*phi(P) = ae*phi(E) + aw*phi(W) + an*phi(N) + as*phi(S) + rhs(P)
!  on rows j=1..n_eta-1 (the outer ring j=n_eta is a fixed Dirichlet boundary
!  and is never written -- its value supplies the N-neighbour of row n_eta-1).
!  E/W are periodic.  The S-neighbour of the wall row (j=1) is ghostS(i); set
!  as(1,:)=0 for a Neumann wall (p'-equation) so ghostS is then irrelevant.
!  Used for u, v and the pressure-correction p'.
! ----------------------------------------------------------------------------
subroutine jacobi5(phi, ap, ae, aw, an, as, rhs, ghostS, omega, nsweep, neta, nxi)
  implicit none
  integer, intent(in) :: neta, nxi, nsweep
  double precision, intent(in) :: omega
  double precision, intent(inout) :: phi(neta, nxi)
  double precision, intent(in) :: ap(neta, nxi), ae(neta, nxi), aw(neta, nxi)
  double precision, intent(in) :: an(neta, nxi), as(neta, nxi), rhs(neta, nxi)
  double precision, intent(in) :: ghostS(nxi)
!f2py intent(in,out) :: phi
  double precision :: pnew(neta, nxi)
  double precision :: pE, pW, pN, pS
  integer :: s, j, i, ip, im
  do s = 1, nsweep
    do i = 1, nxi
      ip = i + 1; if (ip > nxi) ip = 1
      im = i - 1; if (im < 1)   im = nxi
      do j = 1, neta - 1
        pE = phi(j, ip)
        pW = phi(j, im)
        pN = phi(j + 1, i)                       ! row neta is the fixed outer BC
        if (j > 1) then
          pS = phi(j - 1, i)
        else
          pS = ghostS(i)                         ! wall ghost (as(1,:)=0 -> unused)
        end if
        pnew(j, i) = (ae(j,i)*pE + aw(j,i)*pW + an(j,i)*pN + as(j,i)*pS &
                      + rhs(j, i)) / ap(j, i)
      end do
    end do
    do i = 1, nxi
      do j = 1, neta - 1
        phi(j, i) = (1.0d0 - omega) * phi(j, i) + omega * pnew(j, i)
      end do
    end do
  end do
end subroutine jacobi5

! ----------------------------------------------------------------------------
!  mom_coeffs : assemble the implicit momentum-matrix coefficients (shared by u
!  and v) from the lagged face mass fluxes Ff and face diffusivities, using the
!  power-law/upwind convection split a_nb = D_nb + max(-F_nb, 0).
!     Ff* are mass fluxes (rho*u.n*ds) OUTWARD-positive at each face.
!     Df* are diffusion conductances (nu_eff*ds/dn) at each face.
!  Returns ae,aw,an,as and the (un-relaxed) central coefficient apc.
! ----------------------------------------------------------------------------
subroutine mom_coeffs(FfE, FfW, FfN, FfS, DfE, DfW, DfN, DfS, &
                      ae, aw, an, as, apc, neta, nxi)
  implicit none
  integer, intent(in) :: neta, nxi
  double precision, intent(in) :: FfE(neta,nxi), FfW(neta,nxi), FfN(neta,nxi), FfS(neta,nxi)
  double precision, intent(in) :: DfE(neta,nxi), DfW(neta,nxi), DfN(neta,nxi), DfS(neta,nxi)
  double precision, intent(out) :: ae(neta,nxi), aw(neta,nxi), an(neta,nxi), as(neta,nxi)
  double precision, intent(out) :: apc(neta,nxi)
  integer :: j, i
  double precision :: sumF
  do i = 1, nxi
    do j = 1, neta
      ae(j,i) = DfE(j,i) + max(-FfE(j,i), 0.0d0)
      aw(j,i) = DfW(j,i) + max(-FfW(j,i), 0.0d0)
      an(j,i) = DfN(j,i) + max(-FfN(j,i), 0.0d0)
      as(j,i) = DfS(j,i) + max(-FfS(j,i), 0.0d0)
      ! a_P = sum a_nb + net outward flux (the latter -> 0 as continuity converges)
      sumF = FfE(j,i) + FfW(j,i) + FfN(j,i) + FfS(j,i)
      apc(j,i) = ae(j,i) + aw(j,i) + an(j,i) + as(j,i) + sumF
    end do
  end do
end subroutine mom_coeffs
