LIBS_CORE:=util gpio timer system_clock usart

FILTER:=$(foreach LIB,$(CORE_LIBS),%$(LIB)/)
LIBS_CORE:=$(patsubst %,./%/,$(LIBS_CORE))
LIBS:=$(filter-out $(FILTER),$(dir $(shell find $(ls -d ./*/ | sed '/_\/$/d') -maxdepth 1 -name 'Makefile')))

%:
	+$(foreach PATH,$(LIBS_CORE),make -C $(PATH) $@;)
	+$(foreach PATH,$(LIBS),     make -C $(PATH) $@;)
