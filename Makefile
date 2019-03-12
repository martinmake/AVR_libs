LDIR:=$(realpath .)
LADIR:=$(LDIR)/arc
LIDIR:=$(LDIR)/inc
LMDIR:=$(LDIR)/man

CORE_LIBS:=standard usart

export LDIR LADIR LIDIR LMDIR

CORE_LIB_PATHS:=$(foreach CORE_LIB,$(CORE_LIBS),$(dir $(shell find src/$(CORE_LIB) -name Makefile)))
CORE_LIBS     :=$(foreach CORE_LIB,$(CORE_LIBS),%$(CORE_LIB)/) # Make CORE_LIBS filter friendly
LIB_PATHS     :=$(filter-out $(CORE_LIBS)      ,$(dir $(shell find src             -name Makefile)))
EXAMPLE_PATHS :=                                $(dir $(shell find examples        -name Makefile))

.PHONY: all
all:
	+$(foreach PATH,$(CORE_LIB_PATHS)            ,make -e -C $(PATH) -j4;)
	+$(foreach PATH,$(LIB_PATHS) $(EXAMPLE_PATHS),make -e -C $(PATH) -j4;)

.PHONY: clean
clean:
	$(foreach PATH,$(CORE_LIB_PATHS) $(LIB_PATHS) $(EXAMPLE_PATHS),make -e -C $(PATH) clean;)
