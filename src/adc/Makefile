# PROJECT DIRS
IDIRS:=inc
PIDIR:=inc/public
SDIR :=src
ODIR :=obj
DDIR :=dep

# LIB DIRS
LDIR :=$(HOME)/Documents/libs/AVR
LADIR:=$(LDIR)/arc
LIDIR:=$(LDIR)/inc

# TARGETS
MCUS :=atmega328p

# TOOLCHAIN
TOOLCHAIN:=avr-
CC  :=$(TOOLCHAIN)gcc
CXX :=$(TOOLCHAIN)g++
ASMC:=$(TOOLCHAIN)gcc
AR  :=$(TOOLCHAIN)ar
SIZE:=$(TOOLCHAIN)size

# COMPILER FLAGS
WARNING     :=-Wall -Wextra -pedantic -Wno-write-strings
OPTIMIZATION:=-O3 -Os -fdata-sections -ffunction-sections -fno-exceptions

######END#OF#CONFIGURATION#VARIABLES######

LNAME:=$(notdir $(realpath .))

ARC:=$(LADIR)/$(MCU)/lib$(LNAME).a
PHEADERS:=$(shell find $(PIDIR) -name '*.h')
PHEADERS:=$(notdir $(PHEADERS))
PHEADERS:=$(addprefix $(LIDIR)/$(LNAME)/,$(PHEADERS))

C_SRCS  :=$(shell find $(SDIR) -name '*.c'  )
CXX_SRCS:=$(shell find $(SDIR) -name '*.cpp')
ASM_SRCS:=$(shell find $(SDIR) -name '*.S'  )

C_OBJS  :=$(patsubst $(SDIR)/%.c,  $(ODIR)/$(MCU)/%.o,$(C_SRCS)  )
CXX_OBJS:=$(patsubst $(SDIR)/%.cpp,$(ODIR)/$(MCU)/%.o,$(CXX_SRCS))
ASM_OBJS:=$(patsubst $(SDIR)/%.S,  $(ODIR)/$(MCU)/%.o,$(ASM_SRCS))
OBJS:=$(C_OBJS) $(CXX_OBJS) $(ASM_OBJS)
ARC_MEMBERS:=$(notdir $(OBJS))

IDIRS+=$(LIDIR)
IDIRS+=$(PIDIR)
INC:=$(addprefix -I,$(IDIRS))

CCFLAGS :=$(WARNING) $(OPTIMIZATION) -mmcu=$(MCU) $(INC) -std=c99
CXXFLAGS:=$(WARNING) $(OPTIMIZATION) -mmcu=$(MCU) $(INC) -std=c++14
ASMFLAGS:=$(WARNING) $(OPTIMIZATION) -mmcu=$(MCU) $(INC)
ARFLAGS :=rUucv

.SECONDARY: $(OBJS)

.PHONY: all
all:
	+$(foreach MCU,$(MCUS),make -e build MCU=$(MCU);)

.PHONY: build
build: $(ARC)($(ARC_MEMBERS)) $(PHEADERS)
	@echo "> BULDING $(MCU)"
	$(SIZE) $(ARC) | sed -e 's/^/* /'
	@echo "< BULDING $(MCU) DONE"

$(LIDIR)/$(LNAME) $(LADIR)/$(MCU) $(ODIR)/$(MCU):
	mkdir $@

$(ARC)(%.o): $(ODIR)/$(MCU)/%.o | $(LADIR)/$(MCU)
	@echo "> ARCHIVING $<"
	$(AR) $(ARFLAGS) $@ $<
	@echo "< ARCHIVING $< DONE"

$(LIDIR)/$(LNAME)/%.h: $(PIDIR)/%.h | $(LIDIR)/$(LNAME)
	@echo "> STOIRING PUBLIC LIBRARY HEADER $<"
	cp $< -t $(LIDIR)/$(LNAME)
	@echo "< STOIRING PUBLIC LIBRARY HEADER $< DONE"

$(ODIR)/$(MCU)/%.o: $(SDIR)/%.c   | $(ODIR)/$(MCU)
	@echo "> COMPILING $<"
	$(CC)   $(CCFLAGS)  -c $< -o $@
	@echo "< COMPILING $< DONE"
$(ODIR)/$(MCU)/%.o: $(SDIR)/%.cpp | $(ODIR)/$(MCU)
	@echo "> COMPILING $<"
	$(CXX)  $(CXXFLAGS) -c $< -o $@
	@echo "< COMPILING $< DONE"
$(ODIR)/$(MCU)/%.o: $(SDIR)/%.S   | $(ODIR)/$(MCU)
	@echo "> COMPILING $<"
	$(ASMC) $(ASMFLAGS) -c $< -o $@
	@echo "< COMPILING $< DONE"

$(DDIR)/%.d: $(SDIR)/%.c
	@echo "> BUILDING DEPENDENCIES $<"
	$(CC)   $(INC) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/$$(MCU)/%.o,$@)' > $@
	@echo "< BUILDING DEPENDENCIES $< DONE"
$(DDIR)/%.d: $(SDIR)/%.cpp
	@echo "> BUILDING DEPENDENCIES $<"
	$(CXX)  $(INC) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/$$(MCU)/%.o,$@)' > $@
	@echo "< BUILDING DEPENDENCIES $< DONE"
$(DDIR)/%.d: $(SDIR)/%.S
	@echo "> BUILDING DEPENDENCIES $<"
	$(ASMC) $(INC) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/$$(MCU)/%.o,$@)' > $@
	@echo "< BUILDING DEPENDENCIES $< DONE"

.PHONY: clean
clean:
	@echo "> CLEANING"
	-rm -vfr ./$(ODIR)/* ./$(DDIR)/* $(LADIR)/*/lib$(LNAME).a $(LIDIR)/$(LNAME) | sed -e 's/^/* /'
	@echo "< CLEANING DONE"

ifeq ($(NODEP),)
-include $(patsubst $(SDIR)/%.c,  $(DDIR)/%.d,$(C_SRCS)  )
-include $(patsubst $(SDIR)/%.cpp,$(DDIR)/%.d,$(CXX_SRCS))
-include $(patsubst $(SDIR)/%.S,  $(DDIR)/%.d,$(ASM_SRCS))
endif
