LDIR:=$(realpath .)

export LDIR

LIB_PATHS:=$(dir $(shell find src -name Makefile))

.PHONY: all
all:
	make -C src/standard build
	make -C src/usart build
	$(foreach PATH,$(LIB_PATHS),make -e -C $(PATH) -j4 build;)

.PHONY: clean
clean:
	$(foreach PATH,$(LIB_PATHS),make -e -C $(PATH)     clean;)
