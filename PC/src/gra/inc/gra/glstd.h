#ifndef _GRA_GLSTD_H_
#define _GRA_GLSTD_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <inttypes.h>
#include <assert.h>

namespace Gra
{
	inline uint8_t glSizeOf(unsigned int type)
	{
		switch (type)
		{
			case GL_FLOAT:         return 4;
			case GL_UNSIGNED_INT:  return 4;
			case GL_UNSIGNED_BYTE: return 1;
			default: assert(false && "[GRA] glSizeOf: INVALID ENUM");
		}

		return 0;
	}

	enum class DrawMode : GLenum
	{
		TRIANGLES = GL_TRIANGLES,
		LINES     = GL_LINES,
		POINTS    = GL_POINTS,
	};
}

#endif
