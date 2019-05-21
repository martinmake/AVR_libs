LDIR :=$(realpath .)
LLDIR:=$(LDIR)/lib
LIDIR:=$(LDIR)/inc

CORE_LIBS   :=
EXCLUDE_LIBS:=

######END#OF#CONFIGURATION#VARIABLES######

export LDIR LLDIR LIDIR

CORE_LIB_PATHS:=$(foreach LIB,$(CORE_LIBS)   ,            $(dir $(shell find src/$(LIB) -name Makefile)))
CORE_LIBS     :=$(foreach LIB,$(CORE_LIBS)   ,%$(LIB)/)
EXCLUDE_LIBS  :=$(foreach LIB,$(EXCLUDE_LIBS),%$(LIB)/)
LIB_PATHS     :=$(filter-out $(CORE_LIBS) $(EXCLUDE_LIBS),$(dir $(shell find src        -name Makefile)))
EXAMPLE_PATHS :=                                          $(dir $(shell find examples   -name Makefile))

.PHONY: all
all:
	+$(foreach PATH,$(CORE_LIB_PATHS)            ,make -e -C $(PATH);)
	+$(foreach PATH,$(LIB_PATHS) $(EXAMPLE_PATHS),make -e -C $(PATH);)

.PHONY: clean
clean:
	+$(foreach PATH,$(CORE_LIB_PATHS) $(LIB_PATHS) $(EXAMPLE_PATHS),make -e -C $(PATH) clean;)
