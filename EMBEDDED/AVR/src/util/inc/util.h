#ifndef _UTIL_UTIL_H_
#define _UTIL_UTIL_H_

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <avr/io.h>
#include <avr/interrupt.h>
#ifdef F_CPU
#include <util/delay.h>
#endif
#include <util/atomic.h>
#include <avr/pgmspace.h>

#include <math/util.h>

#include "util/assert.h"
#include "util/atomic.h"
#include "util/bits.h"
#include "util/cpp_allocation.h"
#include "util/tty.h"

#endif
