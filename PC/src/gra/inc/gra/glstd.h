#ifndef _GRA_GLSTD_H_
#define _GRA_GLSTD_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <inttypes.h>
#include <assert.h>

#include "gra/gra.h"

namespace Gra
{
	inline uint8_t glSizeOf(unsigned int type)
	{
		switch (type)
		{
			case GL_FLOAT:         return 4;
			case GL_UNSIGNED_INT:  return 4;
			case GL_UNSIGNED_BYTE: return 1;
			default:               assert(false);
		}

		return 0;
	}

	inline GLenum DrawMode_to_GLenum(DrawMode draw_mode)
	{
		switch (draw_mode)
		{
			case DrawMode::TRIANGLES: return GL_TRIANGLES;
			case DrawMode::LINES:     return GL_LINES;
			case DrawMode::POINTS:    return GL_POINTS;
			default:                  assert(false);
		}

		return 0;
	}
}

#endif
