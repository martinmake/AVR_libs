#ifndef _GRA_MATH_H_
#define _GRA_MATH_H_

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Gra
{
	namespace Math
	{
		template<typename T>
		struct vec2
		{
			T x, y;
			vec2(void)                     : x(0),         y(0)         {}
			vec2(T initial_x, T initial_y) : x(initial_x), y(initial_y) {}
			inline vec2& operator+=(const vec2& rhs)       { x += rhs.x; y += rhs.y; return *this; }
			inline vec2  operator- (const vec2& rhs) const { return vec2(x - rhs.x, y - rhs.y); }
			inline vec2  operator+ (const vec2& rhs) const { return vec2(x + rhs.x, y + rhs.y); }
		};
		template<typename T>
		struct vec3
		{
			T x, y, z;
			vec3(void)                                      : x(0),         y(0),         z(0)         {}
			vec3(T initial_x, T initial_y, T initial_z = 0) : x(initial_x), y(initial_y), z(initial_z) {}
		};
		template<typename T>
		struct vec4
		{
			T x, y, z, w;
			vec4(void)                                               : x(0),         y(0),         z(0),         w(0)         {}
			vec4(T initial_x, T initial_y, T initial_z, T initial_w) : x(initial_x), y(initial_y), z(initial_z), w(initial_w) {}
		};
	}
}

#endif
