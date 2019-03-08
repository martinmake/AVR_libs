LDIR:=$(realpath .)

LADIR:=$(LDIR)/arc
LIDIR:=$(LDIR)/inc
LMDIR:=$(LDIR)/man

export LDIR

LIB_PATHS:=$(dir $(shell find src -name Makefile))

.PHONY: all
all:
	-@mkdir -p $(LADIR) $(LIDIR) $(LMDIR)
	make -C src/standard build
	make -C src/usart build
	$(foreach PATH,$(LIB_PATHS),make -C $(PATH) -j4 build;)

.PHONY: clean
clean:
	$(foreach PATH,$(LIB_PATHS),make -C $(PATH)     clean;)
