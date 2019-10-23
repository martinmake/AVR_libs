LIBS_CORE:=math util core

FILTER:=$(foreach LIB,$(CORE_LIBS),%$(LIB)/)
LIBS_CORE:=$(patsubst %,./%/,$(LIBS_CORE))
LIBS:=$(filter-out $(FILTER),$(dir $(shell find $$(ls -d ./*/ | sed '/_\/$$/d') -maxdepth 1 -name 'Makefile')))

.PHONY: all
all:
	+$(foreach PATH,$(LIBS_CORE) $(LIBS),make -C $(PATH) $@;)

%:
	+$(foreach PATH,$(LIBS) $(LIBS_CORE),make -C $(PATH) $@;)
