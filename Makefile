VERBOSE := 1

ifeq ($(VERBOSE),1)
ECHO := 
else
ECHO := @
endif

# Where is the Altera SDK for OpenCL software?
ifeq ($(wildcard $(ALTERAOCLSDKROOT)),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation)
endif
ifeq ($(wildcard $(ALTERAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set ALTERAOCLSDKROOT to the root directory of the Altera SDK for OpenCL software installation.)
endif

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell aocl compile-config )
AOCL_LINK_CONFIG := $(shell aocl link-config )

# Compilation flags
ifeq ($(DEBUG),1)
CXXFLAGS += -g
else
CXXFLAGS += -O2
endif

# Compiler
CXX := g++

#Compiler Flags
CFLAGS := -Wall -c $(DEBUG) -g -DMODEL_DEBUG -DALTERA_CL -DLINUX  
LFLAGS := -Wall $(DEBUG)
MODFLAG := -DM 
PLOT := -DPLOT

#Objects
OBJS := openclBuilder.o options.o opencl.o svd.o timerSample.o util.o 
OBJS_M := main_mod.o

# Target
TARGET := alteraPATHGENnew

# Make it all
all : alteraPATHGENnew

#Building Executable
$(TARGET) : $(OBJS) $(OBJS_M)
	$(CXX) $(OBJS) $(OBJS_M) $(AOCL_COMPILE_CONFIG) -o $(TARGET) $(AOCL_LINK_CONFIG)

# Opencl.cpp
opencl.o : opencl.cpp aocl_utils.h
	$(CXX)  $(AOCL_COMPILE_CONFIG) $(CFLAGS) opencl.cpp

# openclBuilder.cpp
openclBuilder.o : openclBuilder.cpp openclBuilder.h 
	$(CXX) $(MODFLAG)  $(AOCL_COMPILE_CONFIG) $(CFLAGS) openclBuilder.cpp

# option.cpp
#option.o : option.cpp option.h main.h
#	$(CXX) $(CFLAGS) option.cpp

#options.cpp
options.o : options.cpp aocl_utils.h options.h
	$(CXX) $(AOCL_COMPILE_CONFIG) $(CFLAGS) options.cpp

#svd.cpp
svd.o : svd.cpp
	$(CXX) $(CFLAGS) svd.cpp

#timersample.cpp
timerSample.o : timerSample.cpp
	$(CXX)  $(AOCL_COMPILE_CONFIG) $(CFLAGS) timerSample.cpp
	
#utils.cpp
util.o : util.cpp util.h
	$(CXX) -DPLOT  $(AOCL_COMPILE_CONFIG) $(CFLAGS) util.cpp 

#main_mod.cpp
main_mod.o : main_mod.cpp model.h option.h openclBuilder.h openclBuffer.h aocl_utils.h 
	$(CXX) -DPLOT  $(MODFLAG) $(AOCL_COMPILE_CONFIG) $(CFLAGS)  main_mod.cpp

#main.cpp
main.o : main.cpp model.h option.h openclBuilder.h openclBuffer.h aocl_utils.h 
	$(CXX) -DPLOT $(AOCL_COMPILE_CONFIG) $(CFLAGS)  main.cpp
 

 


#Clean
clean:
	rm -f *.o $(TARGET) 
