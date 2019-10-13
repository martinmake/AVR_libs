#ifndef _MATH_UTIL_H_
#define _MATH_UTIL_H_

#include <math.h>

inline int16_t safe_add(int16_t lhs, int16_t rhs)
{
	if (lhs > 0 && rhs > INT16_MAX - lhs) return INT16_MAX;
	if (lhs < 0 && rhs < INT16_MIN - lhs) return INT16_MIN;
	else                                  return lhs + rhs;
}

template <typename T, T low, T high>
T clamp(T value)
{
	static_assert(high > low, "HIGH VALUE HAS TO BE HIGHER THAN LOW VALUE");
	return (value < low) ? low : (value > high) ? high : value;
}

float clamp(float low, float high, float value)
{
	return (value < low) ? low : (value > high) ? high : value;
}

#endif
