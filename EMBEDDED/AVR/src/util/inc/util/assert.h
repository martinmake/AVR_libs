#ifndef _UTIL_ASSERT_H_
#define _UTIL_ASSERT_H_

#include "util/tty.h"

extern void assert_failed(const char* exp, const char* file, int line);
#define assert(exp) if (!(exp)) assert_failed(#exp, __FILE__, __LINE__);
#ifndef CUSTOM_ASSERT
inline void assert_failed(const char* exp, const char* file, int line)
{
	tty_escape_sequence(FG_RED, BOLD);
	fprintf(stderr, "%s:%d: ASSERTION `%s' FAILED\n", file, line, exp);
	tty_escape_sequence(NORMAL);
}
#endif

#endif
