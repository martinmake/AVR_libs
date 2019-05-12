PATHS:=$(dir $(shell find ./*/ -maxdepth 1 -name 'Makefile'))

.PHONY: all
all:
	+$(foreach PATH,$(PATHS),make -e -C $(PATH);)

%:
	+$(foreach PATH,$(PATHS),make -e -C $(PATH) $@;)
