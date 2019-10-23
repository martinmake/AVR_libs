#ifndef _UTIL_ATOMIC_H_
#define _UTIL_ATOMIC_H_

inline bool __sei(void) { sei(); return true;  }
inline bool __cli(void) { cli(); return false; }

#define ATOMIC() for (bool run = __cli(); run; run = __sei())

#endif
