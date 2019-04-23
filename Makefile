LDIR :=$(realpath .)
LADIR:=$(LDIR)/arc
LIDIR:=$(LDIR)/inc

MCUS:=atmega328p

CORE_LIBS   :=standard usart
EXCLUDE_LIBS:=yrobot

######END#OF#CONFIGURATION#VARIABLES######

export LDIR LADIR LIDIR MCUS

CORE_LIB_PATHS:=$(foreach LIB,$(CORE_LIBS)   ,$(dir $(shell find src/$(LIB) -name Makefile)))
CORE_LIBS     :=$(foreach LIB,$(CORE_LIBS)   ,%$(LIB)/) # Make CORE_LIBS filter friendly
EXCLUDE_LIBS  :=$(foreach LIB,$(EXCLUDE_LIBS),%$(LIB)/) # Make CORE_LIBS filter friendly
LIB_PATHS     :=$(filter-out $(CORE_LIBS) $(EXCLUDE_LIBS),$(dir $(shell find src             -name Makefile)))
EXAMPLE_PATHS :=                                          $(dir $(shell find examples        -name Makefile))

.PHONY: all
all:
	+$(foreach PATH,$(CORE_LIB_PATHS)            ,make -e -C $(PATH);)
	+$(foreach PATH,$(LIB_PATHS) $(EXAMPLE_PATHS),make -e -C $(PATH);)

.PHONY: clean
clean:
	+$(foreach PATH,$(CORE_LIB_PATHS) $(LIB_PATHS) $(EXAMPLE_PATHS),make -e -C $(PATH) clean;)
