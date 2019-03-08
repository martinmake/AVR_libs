SDIR :=src
ODIR :=obj
DDIR :=dep
MDIR :=man
LDIR :=$(HOME)/Documents/projects/AVR/libs
LADIR:=$(LDIR)/arc/$(MCU)
LIDIR:=$(LDIR)/inc
LMDIR:=$(LDIR)/man
IDIRS:=inc
PIDIR:=inc/public

LNAME:=$(notdir $(shell realpath .))

MANS :=$(shell find $(MDIR) -name *.*)
CMANS:=$(patsubst $(MDIR)/%,$(LMDIR)/%.gz,$(MANS))

ARC:=$(LADIR)/lib$(LNAME).a
PHEADERS:=$(shell dir $(PIDIR)/*.h)
PHEADERS:=$(notdir $(PHEADERS))
PHEADERS:=$(addprefix $(LIDIR)/$(LNAME)/,$(PHEADERS))

C_SRCS  :=$(shell find $(SDIR) -name *.c)
CXX_SRCS:=$(shell find $(SDIR) -name *.cpp)
ASM_SRCS:=$(shell find $(SDIR) -name *.S)

C_OBJS  :=$(patsubst $(SDIR)/%.c,$(ODIR)/$(MCU)/%.o,$(C_SRCS))
CXX_OBJS:=$(patsubst $(SDIR)/%.cpp,$(ODIR)/$(MCU)/%.o,$(CXX_SRCS))
ASM_OBJS:=$(patsubst $(SDIR)/%.S,$(ODIR)/$(MCU)/%.o,$(ASM_SRCS))
OBJS:=$(C_OBJS) $(CXX_OBJS) $(ASM_OBJS)

ifeq ($(strip $(OBJS)),)
	ARC:=
endif

IDIRS+=$(LIDIR)
IDIRS+=$(PIDIR)
INC:=$(patsubst %,-I%,$(IDIRS))

CC  :=avr-gcc
CXX :=avr-g++
ASMC:=avr-gcc
AR  :=ar

WARNING :=-Wall -Wextra -pedantic -Wno-write-strings
CCFLAGS :=$(WARNING) -Os -mmcu=$(MCU) $(INC) -std=c99
CXXFLAGS:=$(WARNING) -Os -mmcu=$(MCU) $(INC) -std=c++14
ASMFLAGS:=$(WARNING) -Os -mmcu=$(MCU) $(INC)
ARFLAGS :=rUucv

.PHONY: all
all: $(ARC)($(OBJS)) $(CMANS) $(PHEADERS)

$(ARC)(%.o): %.o $(LADIR)
	@echo =======ARCHIVING=========
	$(AR) $(ARFLAGS) $@ $<
	@echo ==========DONE===========

$(LIDIR)/$(LNAME)/%.h: $(PIDIR)/%.h $(LIDIR)/$(LNAME)
	@echo ==BUILDING=LIB=HEADERS===
	cp $< -t $(LIDIR)/$(LNAME)
	@echo ==========DONE===========

$(LIDIR)/$(LNAME):
	@echo ====LIB=INC=DIR=BUILD====
	mkdir $@
	@echo ==========DONE===========

$(LADIR):
	@echo ====LIB=ARC=DIR=BUILD====
	mkdir $@
	@echo ==========DONE===========

$(ODIR)/$(MCU):
	@echo ====MCU=OBJ=DIR=BUILD====
	mkdir $@
	@echo ==========DONE===========

$(LMDIR)/%.gz: $(MDIR)/%
	@echo ===BUILDING=MAN=PAGES====
	gzip $< -c > $@
	@echo ==========DONE===========

$(ODIR)/$(MCU)/%.o: $(SDIR)/%.c $(ODIR)/$(MCU)
	@echo =======COMPILING=========
	$(CC) $(CCFLAGS) -c $< -o $@
	@echo ==========DONE===========

$(ODIR)/$(MCU)/%.o: $(SDIR)/%.cpp $(ODIR)/$(MCU)
	@echo =====COMPILING======
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo ==========DONE===========

$(ODIR)/$(MCU)/%.o: $(SDIR)/%.S $(ODIR)/$(MCU)
	@echo =======COMPILING=========
	$(ASMC) $(ASMFLAGS) -c $< -o $@
	@echo ==========DONE===========

$(DDIR)/%.d: $(SDIR)/%.c
	@echo ==BUILDING=DEPENDENCIES==
	$(CC) $(INC) -MM $< -MT $(patsubst $(DDIR)/%.d,$(ODIR)/$$\(MCU\)/%.o,$@) > $@
	@echo ==========DONE===========

$(DDIR)/%.d: $(SDIR)/%.cpp
	@echo ==BUILDING=DEPENDENCIES==
	$(CXX) $(INC) -MM $< -MT $(patsubst $(DDIR)/%.d,$(ODIR)/$$\(MCU\)/%.o,$@) > $@
	@echo ==========DONE===========

$(DDIR)/%.d: $(SDIR)/%.S
	@echo ==BUILDING=DEPENDENCIES==
	$(ASMC) $(INC) -mmcu=$(MCU) -MM $< -MT $(patsubst $(DDIR)/%.d,$(ODIR)/$$\(MCU\)/%.o,$@) > $@
	@echo ==========DONE===========

.PHONY: build
build:
	@echo =========BULDING=========
	@./build
	@echo ==========DONE===========

.PHONY: clean
clean:
	@echo ========CLEANING=========
	-rm -rfv $(ODIR)/* $(DDIR)/* $(LADIR)/*/lib$(LNAME).a $(LIDIR)/$(LNAME) $(CMANS)
	@echo ==========DONE===========

-include $(patsubst $(SDIR)/%.c,$(DDIR)/%.d,$(C_SRCS))
-include $(patsubst $(SDIR)/%.cpp,$(DDIR)/%.d,$(CXX_SRCS))
-include $(patsubst $(SDIR)/%.S,$(DDIR)/%.d,$(ASM_SRCS))
