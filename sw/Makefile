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

#Objects
OBJS := util.o svd.o main.o
#OBJS_M := main_mod.o

# Target
TARGET := alteraPATHGENnewSW

# Make it all
all : alteraPATHGENnewSW

#Building Executable
$(TARGET) : $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET)

# util.cpp
util.o : util.cpp util.h
	$(CXX)  $(CFLAGS) util.cpp

# svd.cpp
svd.o : svd.cpp  
	$(CXX)   $(CFLAGS) svd.cpp

#main.cpp
main.o : main.cpp main.h util.h 
	$(CXX)  $(CFLAGS)  main.cpp
 

 


#Clean
clean:
	rm -f *.o $(TARGET) 
