#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Random.H>
#include <AMReX_Print.H>
#include <AMReX_FFT.H>
#include <AMReX_Vector.H>
//#include <AMReX_PlotFileUtilHDF5.H>

using namespace amrex;

const int N = 32;
const amrex::Real L = 3000.;

/****
    Small functions (less than 10 lines)
****/

// Nyquist condition
int flip_index(int indx) { return std::abs(N - indx); }

// Nyquist condition and calculation of kmag
int invert_index(int indx) { return (int)(N/2 - std::abs(N/2 - indx)); }

/****
    Initialisation routines
****/

// Returns analytic power spectrum in modulus/argument form
GpuComplex<Real> calculate_mode_function(double km, std::string spec_type)
{
    const Real Mp = 1.;
    const Real m = 1.e-4;
    const Real phi0 = 4.;
    const Real Pi0 = -0.00001627328880468423;

    // Deals with k=0 case, which is undefined if m=0
    if(km < 1.e-23) { return 0.; }
    
    // Stores modulus and argument 
    Real ms_mag = 0.;
    Real ms_arg = 0.;

    // Hubble at t=0, needed for tensor solution
    Real H0 = sqrt((4.0 * M_PI/3.0/pow(Mp, 2.)) * (pow(m * phi0, 2.0) + pow(Pi0, 2.)));

    double kpr = km/H0;
    if (spec_type == "position") // Position mode funcion
    {
        ms_mag = sqrt((1.0/km + H0*H0/pow(km, 3.))/2.);
        ms_arg = atan2((cos(kpr) + kpr*sin(kpr)), (kpr*cos(kpr) - sin(kpr)));
    }
    else if (spec_type == "velocity") // Velocity mode funcion
    {
        ms_mag = sqrt(km/2.);
        ms_arg = -atan2(cos(kpr), sin(kpr));
    }
    else { Error("RandomField::calculate_mode_function Value of spec_type not allowed."); }

    // Construct the mode function and return it
    GpuComplex<Real> ps(ms_mag * cos(ms_arg), ms_mag * sin(ms_arg));
    return ps;
}

// Turns analytic PS into GRF and applies window function if requested
GpuComplex<Real> calculate_random_field(IntVect iv, std::string spectrum_type, 
                                            Real rand_amp, Real rand_phase)
{
    const Real kstar = 24.;
    const Real Delta = 30.;

    GpuComplex<Real> value(0., 0.);

    // Find kmag with FFTW-style inversion on the last two indices
    int i = iv[0];
    int j = invert_index(iv[1]);
    int k = invert_index(iv[2]);

    double kmag = std::sqrt(i*i + j*j + k*k) * 2 * M_PI / L;

    // Find the analytic power spectrum
    value = calculate_mode_function(kmag, spectrum_type);

    // Make one random draw for the amplitude and phase individually
    Real rand_mod = sqrt(-2. * log(rand_amp)); // Rayleigh distribution about |h|
    Real rand_arg = 2. * M_PI * rand_phase;      // Uniform random phase

    // Multiply amplitude by Rayleigh draw
    value *= rand_mod;

    // Apply the random phase (assuming MS phase is accounted for)
    Real new_real = value.real() * cos(rand_arg) - value.imag() * sin(rand_arg);
    Real new_imag = value.real() * sin(rand_arg) + value.imag() * cos(rand_arg);
    GpuComplex<Real> new_value(new_real, new_imag);

    value = new_value;

    // Apply a window function if requested
    double ks = kstar * 2. * M_PI/L;
    double Dt = L/Delta;
    value *= 0.5 * (1.0 - tanh(Dt * (kmag - ks))); 

    return value;
}


// Applies above Nyquist conditions to a given MF
 void apply_nyquist_conditions(cMultiFab &field)
{
    int nc = field.nComp();
    for (MFIter mfi(field); mfi.isValid(); ++mfi) 
    {
        // The geometry for this MPI rank
        const Box& bx = mfi.fabbox();
        Array4<GpuComplex<Real>> const& field_ptr = field.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            IntVect iv = {i, j, k};

            if ((i == 0 || i == N/2) && (j == 0 || j == N/2) && (k == 0 || k == N/2))
            {
                for(int comp = 0; comp < nc; comp++)
                {
                    GpuComplex<Real> temp(field_ptr(i, j, k, comp).real(), 0.);
                    field_ptr(i, j, k, comp) = temp;
                }
            }

            else if (i==0 || i==N/2) 
            {
                if((k > N/2 && j == N/2) || (k == 0 && j > N/2) ||
                    (k > N/2 && j == 0) || (k == N/2 && j > N/2))
                {
                    for(int comp = 0; comp < nc; comp++) 
                    {
                        GpuComplex<Real> temp(field_ptr(i, invert_index(j), invert_index(k), comp).real(), 
                                                -field_ptr(i, invert_index(j), invert_index(k), comp).imag());
                        field_ptr(i, j, k, comp) = temp;
                    }
                }
                
                else if(j > N/2)
                {
                    for(int comp = 0; comp < nc; comp++) 
                    {
                        GpuComplex<Real> temp(field_ptr(i, invert_index(j), flip_index(k), comp).real(), 
                                                -field_ptr(i, invert_index(j), flip_index(k), comp).imag());
                        field_ptr(i, j, k, comp) = temp;
                    }
                }
            }
        });
    }
}