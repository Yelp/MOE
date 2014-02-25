include ../../../../OL_ARCH
#if this does not exist, run "configure" in the root directory

# TODO(eliu): clean up these options: which are needed, what (remaining) is redundant with -O2 (on gcc & icc), etc (#40473)
CFLAGS_GCC = -std=c++11 -Wold-style-cast -Wnon-virtual-dtor -Wctor-dtor-privacy -Woverloaded-virtual -Wsign-promo -Wundef -Wshadow -Wcast-align -Wzero-as-null-pointer-constant -Wall -Wextra -g -O2 -fopenmp -march=native -fPIC -ftree-vectorize -fomit-frame-pointer -fno-trapping-math -fno-signaling-nans -fno-math-errno #-fassociative-math
CFLAGS_ICC = $(CFLAGS_GCC)
#CFLAGS_ICC += -vec-report2 -cxxlib -fstrict-aliasing -alias-const -freorder-blocks -ftz -no-prec-div -no-prec-sqrt -fast-transcendentals -fp-model fast=1 -vec-guard-write -align -finline -restrict -ip -opt-multi-version-aggressive -falign-stack=maintain-16-byte -opt-streaming-stores auto -opt-subscript-in-range

#CFLAGS = $(CFLAGS_ICC)
CFLAGS = $(CFLAGS_GCC)
CFLAGS += $(CXXOPTS)
#CFLAGS = -Wall -g -O0 -std=gnu++11 -fopenmp

#enable debug printing
#CFLAGS += -DOL_DEBUG_PRINT=1
# enable warning/error printing
CFLAGS += -DOL_WARNING_PRINT=1

CFLAGS_TEST = -Wall -g -O0 -std=gnu++11 -fopenmp

# the switch to tell the linker to remove dead code is different on OSX & Linux
UNAME_S = $(shell uname -s)
LINKFLAGS =
ifeq ($(UNAME_S),Darwin)
	LINKFLAGS += -Wl,-dead_strip
endif
ifeq ($(UNAME_S),Linux)
	LINKFLAGS += -Wl,--gc-sections #-print-gc-sections #-static
endif

# TODO(eliu): Clean up the way include dirs are resolved; hard-coding is terrible (#40473)
INCLUDE = -I/nail/home/eliu/local/include $(INCEXT)
#INCLUDE = -I/opt/local/include -I/usr/include/python$(PYTHON_VER)
DYNLIBS = -L/opt/local/lib -l$(BOOST_PYTHON) -lpython$(PYTHON_VER)

INC = .

DEPENDENCIES =   $(INC)/gpp_common.hpp \
	             $(INC)/gpp_covariance.hpp \
		 $(INC)/gpp_covariance_test.hpp \
		 $(INC)/gpp_domain.hpp \
		 $(INC)/gpp_domain_test.hpp \
		 $(INC)/gpp_exception.hpp \
		 $(INC)/gpp_geometry.hpp \
		 $(INC)/gpp_geometry_test.hpp \
	         $(INC)/gpp_linear_algebra.hpp \
	         $(INC)/gpp_linear_algebra-inl.hpp \
		 $(INC)/gpp_linear_algebra_test.hpp \
		 $(INC)/gpp_logging.hpp \
	         $(INC)/gpp_math.hpp \
	         $(INC)/gpp_math-inl.hpp \
		 $(INC)/gpp_math_test.hpp \
		 $(INC)/gpp_mock_optimization_objective_functions.hpp \
	         $(INC)/gpp_model_selection_and_hyperparameter_optimization.hpp \
		 $(INC)/gpp_model_selection_and_hyperparameter_optimization_test.hpp \
		 $(INC)/gpp_optimization.hpp \
		 $(INC)/gpp_optimization_parameters.hpp \
		 $(INC)/gpp_optimization_test.hpp \
		 $(INC)/gpp_random.hpp \
		 $(INC)/gpp_random_test.hpp \
		 $(INC)/gpp_test_utils.hpp

SRCS =   $(INC)/gpp_covariance.cpp \
		 $(INC)/gpp_covariance_test.cpp \
		 $(INC)/gpp_domain_test.cpp \
	     $(INC)/gpp_exception.cpp \
		 $(INC)/gpp_geometry_test.cpp \
	     $(INC)/gpp_linear_algebra.cpp \
		 $(INC)/gpp_linear_algebra_test.cpp \
		 $(INC)/gpp_logging.cpp \
	     $(INC)/gpp_math.cpp \
		 $(INC)/gpp_math_test.cpp \
	     $(INC)/gpp_model_selection_and_hyperparameter_optimization.cpp \
		 $(INC)/gpp_model_selection_and_hyperparameter_optimization_test.cpp \
		 $(INC)/gpp_optimization_test.cpp \
		 $(INC)/gpp_random.cpp \
		 $(INC)/gpp_random_test.cpp \
		 $(INC)/gpp_test_utils.cpp

OBJS = $(SRCS:.cpp=.o)

CFLAGS_NOPRINT = $(filter-out -DOL_DEBUG_PRINT%, $(CFLAGS))

.PHONY:
	echo $(CFLAGS_NOPRINT)

default: shared

clean:
	rm *.o *.so EPI demo_full demo_EI demo_hyper

%.o: %.cpp $(DEPENDENCIES)
	$(CXX) $(CFLAGS) -c $(INCLUDE) $< -o $@

EPI: $(DEPENDENCIES) $(OBJS) gpp_core.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o EPI gpp_core.o $(OBJS) $(LLIBS) -lm $(LINKFLAGS)

shared: $(DEPENDENCIES) $(OBJS) gpp_python.o
	$(CXX) $(CFLAGS) $(INCLUDE) -shared -o GPP.so gpp_python.o $(OBJS) $(LLIBS) -lm $(LINKFLAGS)

demo_EI: $(DEPENDENCIES) gpp_covariance.o gpp_exception.o gpp_linear_algebra.o gpp_math.o gpp_model_selection_and_hyperparameter_optimization.o gpp_random.o gpp_expected_improvement_demo.o
	$(CXX) $(CFLAGS) $(INCLUDE) gpp_covariance.o gpp_exception.o gpp_linear_algebra.o gpp_math.o gpp_model_selection_and_hyperparameter_optimization.o gpp_random.o gpp_expected_improvement_demo.o -o demo_EI $(LLIBS) -lm $(LINKFLAGS)

demo_hyper: $(DEPENDENCIES) gpp_covariance.o gpp_exception.o gpp_linear_algebra.o gpp_math.o gpp_model_selection_and_hyperparameter_optimization.o gpp_random.o gpp_hyperparameter_optimization_demo.o
	$(CXX) $(CFLAGS) $(INCLUDE) gpp_covariance.o gpp_exception.o gpp_linear_algebra.o gpp_math.o gpp_model_selection_and_hyperparameter_optimization.o gpp_random.o gpp_hyperparameter_optimization_demo.o -o demo_hyper $(LLIBS) -lm $(LINKFLAGS)

demo_full: $(DEPENDENCIES) gpp_covariance.o gpp_exception.o gpp_linear_algebra.o gpp_math.o gpp_model_selection_and_hyperparameter_optimization.o gpp_random.o gpp_hyper_and_EI_demo.o
	$(CXX) $(CFLAGS) $(INCLUDE) gpp_covariance.o gpp_exception.o gpp_linear_algebra.o gpp_math.o gpp_model_selection_and_hyperparameter_optimization.o gpp_random.o gpp_hyper_and_EI_demo.o -o demo_full $(LLIBS) -lm $(LINKFLAGS)

demos: demo_EI demo_hyper demo_full
