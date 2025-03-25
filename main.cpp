#include "functions.hpp"

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv); {
    
    amrex::InitRandom(1357812);

    // Make the Fourier transform and derive the Fourier space MF ingredients
    IntVect domain_low(0, 0, 0);
    IntVect domain_high(N-1, N-1, N-1);
    Box x_domain(domain_low, domain_high);
    BoxArray xba(x_domain);
    DistributionMapping xdm(xba);
    FFT::R2C<Real> my_fft(x_domain);

    // Set up the problem domain in Fourier space
    // And impose that MPI ranks only slice along the i index (for Nyquist conditions)
    IntVect k_domain_high(N/2, N-1, N-1);
    Box k_domain(domain_low, k_domain_high);
    Array< bool, AMREX_SPACEDIM > const &slicing{true, false, false};
    BoxArray kba = decompose(k_domain, ParallelContext::NProcsAll(), slicing);
    DistributionMapping kdm(kba);

    // Set up the MFs to store the in/out data sets
    cMultiFab hs_k(kba, kdm, 2, 0);
    MultiFab hs_x(xba, xdm, 2, 0);

    for (MFIter mfi(hs_k); mfi.isValid(); ++mfi) 
    {
        const Box& bx = mfi.fabbox();
        Array4<GpuComplex<Real>> const& hs_ptr = hs_k.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            IntVect iv = {i, j, k};
            for(int p=0; p<2; p++)
            {
                Real draw1 = amrex::Random();
                Real draw2 = amrex::Random();

                // See functions.hpp
                hs_ptr(i, j, k, p) = calculate_random_field(iv, "position", draw1, draw2);
            }
        });
    }

    apply_nyquist_conditions(hs_k); // functions.hpp

    for(int fcomp = 0; fcomp < hs_k.nComp(); fcomp++)
    {
        cMultiFab hs_k_slice(hs_k, make_alias, fcomp, 1);
        MultiFab hs_x_slice(hs_x, make_alias, fcomp, 1);
        my_fft.backward(hs_k_slice, hs_x_slice);
    }

    hs_x.mult(pow(2. * M_PI/L, 3.));

    std::string filename = "./mode-function";
    Vector<std::string> field_names{"plus", "cross"};

    WriteSingleLevelPlotfileHDF5(filename, hs_x, field_names);

    } amrex::Finalize();
}