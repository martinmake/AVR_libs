PATHS:=$(dir $(shell find $$(ls -d ./*/ | sed '/_\/$$/d') -maxdepth 1 -name 'Makefile'))

.PHONY: all
all:
	+$(foreach PATH,$(PATHS),make -C $(PATH) $@;)

%:
	+$(foreach PATH,$(PATHS),make -C $(PATH) $@;)
