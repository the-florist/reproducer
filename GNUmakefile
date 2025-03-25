AMREX_HOME ?= ../amrex

DEBUG = TRUE
DIM = 3
COMP = intel-llvm
TINY_PROFILE = FALSE

USE_MPI = FALSE
USE_CUDA = FALSE
USE_HIP = FALSE

USE_FFT = TRUE
USE_HDF5 = TRUE

BL_NO_FORT = TRUE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/FFT/Make.package
include $(AMREX_HOME)/Src/Extern/HDF5/Make.package
include Make.package


include $(AMREX_HOME)/Tools/GNUMake/Make.rules