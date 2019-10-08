PATHS:=$(dir $(shell find $(ls -d ./*/ | sed '/_\/$/d') -maxdepth 1 -name 'Makefile'))

%:
	+$(foreach PATH,$(PATHS),make -C $(PATH) $@;)
