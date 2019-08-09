#ifndef _GRA_GLSTD_H_
#define _GRA_GLSTD_H_

#include <GL/glew.h>

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
			default:               assert(false);
		}

		return 0;
	}
}

#endif
