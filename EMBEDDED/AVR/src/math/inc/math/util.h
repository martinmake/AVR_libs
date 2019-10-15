#ifndef _MATH_UTIL_H_
#define _MATH_UTIL_H_

#include <math.h>

inline int16_t safe_add(int16_t lhs, int16_t rhs)
{
	if (lhs > 0 && rhs > INT16_MAX - lhs) return INT16_MAX;
	if (lhs < 0 && rhs < INT16_MIN - lhs) return INT16_MIN;
	else                                  return lhs + rhs;
}

template <typename output_t, typename input_t>
output_t safe_cast(input_t value)
{
	static_assert(false && sizeof(output_t), "TYPES NOT SUPPORTED");
}
template <>
inline uint8_t safe_cast<uint8_t>(int16_t value)
{
	if (value > UINT8_MAX) return UINT8_MAX;
	if (value <         0) return         0;
	else                   return value;
}

template <typename T, T low, T high>
T clamp(T value)
{
	static_assert(high > low, "HIGH VALUE HAS TO BE HIGHER THAN LOW VALUE");
	return (value < low) ? low : (value > high) ? high : value;
}

inline float clamp(float low, float high, float value)
{
	return (value < low) ? low : (value > high) ? high : value;
}

#endif
