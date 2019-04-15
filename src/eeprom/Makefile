# PROJECT DIRS
SDIR :=src
ODIR :=obj
DDIR :=dep
MDIR :=man
IDIRS:=inc
PIDIR:=inc/public

# LIB DIRS
LDIR :=$(HOME)/Documents/proj/AVR/libs
LADIR:=$(LDIR)/arc
LIDIR:=$(LDIR)/inc
LMDIR:=$(LDIR)/man

# MCU
MCUS :=atmega328p

# TOOLCHAIN
TOOLCHAIN:=avr-
CC  :=$(TOOLCHAIN)gcc
CXX :=$(TOOLCHAIN)g++
ASMC:=$(TOOLCHAIN)gcc
AR  :=$(TOOLCHAIN)ar

# COMPILER FLAGS
WARNING :=-Wall -Wextra -pedantic -Wno-write-strings
OPTIMIZATION:=-O3 -Os -fdata-sections -ffunction-sections -fno-exceptions

######END#OF#CONFIGURATION#VARIABLES######

LNAME:=$(notdir $(realpath .))

MANS :=$(shell find $(MDIR) -name '*.*' 2>/dev/null)
CMANS:=$(patsubst $(MDIR)/%,$(LMDIR)/%.gz,$(MANS)  )

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
OBJS    :=$(C_OBJS) $(CXX_OBJS) $(ASM_OBJS)

ARC_MEMBERS:=$(notdir $(OBJS))

IDIRS+=$(PIDIR)
INC  :=$(addprefix -I,$(IDIRS) $(LIDIR))

CCFLAGS :=$(WARNING) $(OPTIMIZATION) -mmcu=$(MCU) $(INC) -std=c99
CXXFLAGS:=$(WARNING) $(OPTIMIZATION) -mmcu=$(MCU) $(INC) -std=c++14
ASMFLAGS:=$(WARNING) $(OPTIMIZATION) -mmcu=$(MCU) $(INC)
ARFLAGS :=rUucv

.SECONDARY: $(OBJS)

.PHONY: all
all:
	@echo "> BULDING"
	+$(foreach MCU,$(MCUS),make -e build MCU=$(MCU);)
	@echo "< BUILDING DONE"

.PHONY: build
build: $(ARC)($(ARC_MEMBERS)) $(CMANS) $(PHEADERS)

$(LIDIR)/$(LNAME) $(LADIR)/$(MCU) $(ODIR)/$(MCU):
	mkdir $@

$(ARC)(%.o): $(ODIR)/$(MCU)/%.o | $(LADIR)/$(MCU)
	@echo "> ARCHIVING $<"
	$(AR) $(ARFLAGS) $@ $<
	@echo "< ARCHIVING $< DONE"

$(LIDIR)/$(LNAME)/%.h: $(PIDIR)/%.h | $(LIDIR)/$(LNAME)
	@echo "> BUILDING LIBRARY HEADERS"
	cp $< -t $(LIDIR)/$(LNAME)
	@echo "< BUILDING LIBRARY HEADERS DONE"

$(LMDIR)/%.gz: $(MDIR)/%
	@echo "> BUILDING MAN PAGES"
	gzip $< -c > $@
	@echo "< BUILDING MAN PAGES DONE"

$(ODIR)/$(MCU)/%.o: $(SDIR)/%.c | $(ODIR)/$(MCU)
	@echo "> COMPILING: $<"
	$(CC)   $(CCFLAGS) -c $< -o $@
	@echo "< COMPILING: $< DONE"
$(ODIR)/$(MCU)/%.o: $(SDIR)/%.cpp | $(ODIR)/$(MCU)
	@echo "> COMPILING: $<"
	$(CXX)  $(CXXFLAGS) -c $< -o $@
	@echo "< COMPILING: $< DONE"
$(ODIR)/$(MCU)/%.o: $(SDIR)/%.S | $(ODIR)/$(MCU)
	@echo "> COMPILING: $<"
	$(ASMC) $(ASMFLAGS) -c $< -o $@
	@echo "< COMPILING: $< DONE"

$(DDIR)/%.d: $(SDIR)/%.c
	@echo "> BUILDING DEPENDENCY: $<"
	$(CC)   $(INC) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/$$(MCU)/%.o,$@)' > $@
	@echo "< BUILDING DEPENDENCY: $< DONE"
$(DDIR)/%.d: $(SDIR)/%.cpp
	@echo "> BUILDING DEPENDENCY: $<"
	$(CXX)  $(INC) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/$$(MCU)/%.o,$@)' > $@
	@echo "< BUILDING DEPENDENCY: $< DONE"
$(DDIR)/%.d: $(SDIR)/%.S
	@echo "> BUILDING DEPENDENCY: $<"
	$(ASMC) $(INC) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/$$(MCU)/%.o,$@)' > $@
	@echo "< BUILDING DEPENDENCY: $< DONE"

.PHONY: setup
setup:
	@echo "> MAKING DIRECTORIES"
	mkdir -p $(ODIR) $(DDIR) $(IDIRS) $(SDIR)
	touch $(ODIR)/.empty $(DDIR)/.empty
	@echo "< MAKING DIRECTORIES DONE"

.PHONY: clean
clean:
	@echo "> CLEANING"
	-rm -vfr ./$(ODIR)/* ./$(DDIR)/* $(LADIR)/*/lib$(LNAME).a $(LIDIR)/$(LNAME) $(CMANS)
	@echo "< CLEANING DONE"

ifeq ($(NODEP),)
-include $(patsubst $(SDIR)/%.c,  $(DDIR)/%.d,$(C_SRCS)  )
-include $(patsubst $(SDIR)/%.cpp,$(DDIR)/%.d,$(CXX_SRCS))
-include $(patsubst $(SDIR)/%.S,  $(DDIR)/%.d,$(ASM_SRCS))
endif
