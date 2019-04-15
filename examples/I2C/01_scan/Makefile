IDIRS:=inc
SDIR :=src
ODIR :=obj
DDIR :=dep
BDIR :=bin
ELF  :=$(BDIR)/main.elf
HEX  :=$(BDIR)/main.hex
BIN  :=$(BDIR)/main.bin
ASM  :=$(BDIR)/main.asm
LDIR :=$(HOME)/Documents/proj/AVR/libs
LIDIR:=$(LDIR)/inc
LADIR:=$(LDIR)/arc

MCU  :=atmega328p
F_CPU:=16000000
BAUD :=1000000
PROG :=arduino
PORT :=/dev/ttyUSB0

LIBS:=usart i2c
DEFS:=

######END#OF#CONFIGURATION#VARIABLES######

LIBS+=standard
DEFS+=$(EDEFS) F_CPU=$(F_CPU) BAUD=$(BAUD)

LIBS  :=$(addprefix -l,$(LIBS))
DEFS  :=$(addprefix -D,$(DEFS))

C_SRCS  :=$(shell find $(SDIR) -name *.c  )
CXX_SRCS:=$(shell find $(SDIR) -name *.cpp)
ASM_SRCS:=$(shell find $(SDIR) -name *.S  )

C_OBJS  :=$(patsubst $(SDIR)/%.c,  $(ODIR)/%.o,$(C_SRCS)  )
CXX_OBJS:=$(patsubst $(SDIR)/%.cpp,$(ODIR)/%.o,$(CXX_SRCS))
ASM_OBJS:=$(patsubst $(SDIR)/%.S,  $(ODIR)/%.o,$(ASM_SRCS))
OBJS:=$(C_OBJS) $(CXX_OBJS) $(ASM_OBJS)

IDIRS+=$(LIDIR)
INC:=$(addprefix -I,$(IDIRS))

CC  :=avr-gcc
CXX :=avr-g++
ASMC:=avr-gcc
LD  :=avr-g++

WARNING :=-Wall -Wextra -pedantic -Wno-write-strings
CCFLAGS :=$(WARNING) -Os -mmcu=$(MCU) $(INC)  $(DEFS) -std=c99
CXXFLAGS:=$(WARNING) -Os -mmcu=$(MCU) $(INC)  $(DEFS) -std=c++14
ASMFLAGS:=$(WARNING) -Os -mmcu=$(MCU) $(INC)  $(DEFS)
LDFLAGS :=$(WARNING) -Os -mmcu=$(MCU) -L$(LADIR)/$(MCU)

.SECONDARY: $(OBJS)

.PHONY: all
all: $(ELF)
	@echo =========BULDING=========
	avr-objcopy -j .data -j .text -O ihex   $(ELF) $(HEX)
	avr-objcopy -j .data -j .text -O binary $(ELF) $(BIN)
	avr-objdump -Dz $(ELF) > $(ASM)
	avr-size $(HEX)
	@echo ==========DONE===========

$(ELF): $(OBJS)
	@echo ========LINKING==========
	$(LD) $(LDFLAGS) $^ $(LIBS) -o $@
	@echo ==========DONE===========

$(ODIR)/%.o: $(SDIR)/%.c
	@echo =======COMPILING=========
	$(CC)   $(CCFLAGS) -c $< -o $@
	@echo ==========DONE===========
$(ODIR)/%.o: $(SDIR)/%.cpp
	@echo =======COMPILING=========
	$(CXX)  $(CXXFLAGS) -c $< -o $@
	@echo ==========DONE===========
$(ODIR)/%.o: $(SDIR)/%.S
	@echo =======COMPILING=========
	$(ASMC) $(ASMFLAGS) -c $< -o $@
	@echo ==========DONE===========

$(DDIR)/%.d: $(SDIR)/%.c
	@echo ==BUILDING=DEPENDENCIES==
	$(CC)   $(INC) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/%.o,$@)' > $@
	@echo ==========DONE===========
$(DDIR)/%.d: $(SDIR)/%.cpp
	@echo ==BUILDING=DEPENDENCIES==
	$(CXX)  $(INC) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/%.o,$@)' > $@
	@echo ==========DONE===========
$(DDIR)/%.d: $(SDIR)/%.S
	@echo ==BUILDING=DEPENDENCIES==
	$(ASMC) $(INC) -MM $< -MT '$(patsubst $(DDIR)/%.d,$(ODIR)/%.o,$@)' > $@
	@echo ==========DONE===========

.PHONY: flash
flash:
	sudo avrdude -p $(MCU) -c $(PROG) -P $(PORT) -U flash:w:$(HEX):i

.PHONY: serial
serial:
	sudo tio -b $(BAUD) -d 8 -s 2 -e -mINLCRNL $(PORT)

.PHONY: setup
setup:
	@echo ===MAKING=DIRECTORIES====
	mkdir -p $(BDIR) $(ODIR) $(DDIR) $(IDIRS) $(SDIR)
	touch $(BDIR)/.empty $(ODIR)/.empty $(DDIR)/.empty
	@echo ==========DONE===========

.PHONY: clean
clean:
	@echo ========CLEANING=========
	-rm -vfr ./$(ODIR)/* ./$(DDIR)/* ./$(BDIR)/*
	@echo ==========DONE===========

-include $(patsubst $(SDIR)/%.c,  $(DDIR)/%.d,$(C_SRCS)  )
-include $(patsubst $(SDIR)/%.cpp,$(DDIR)/%.d,$(CXX_SRCS))
-include $(patsubst $(SDIR)/%.S,  $(DDIR)/%.d,$(ASM_SRCS))
