#ifndef _GRA_GLDEBUG_H_
#define _GRA_GLDEBUG_H_

#include <iostream>
#include <assert.h>
#include <stdio.h>

#include "gra/glstd.h"

namespace Gra
{
	inline void glClearError()
	{
		while (glGetError() != GL_NO_ERROR) {}
	}

	inline bool glLogCall()
	{
		bool found_error = false;

		while (GLenum error = glGetError())
		{
			std::cerr << "[GL_ERROR] ";
			switch (error)
			{
				case GL_INVALID_ENUM:      std::cerr << "GL_INVALID_ENUM";      break;
				case GL_INVALID_VALUE:     std::cerr << "GL_INVALID_VALUE";     break;
				case GL_INVALID_OPERATION: std::cerr << "GL_INVALID_OPERATION"; break;
				case GL_STACK_OVERFLOW:    std::cerr << "GL_STACK_OVERFLOW";    break;
				case GL_STACK_UNDERFLOW:   std::cerr << "GL_STACK_UNDERFLOW";   break;
				case GL_OUT_OF_MEMORY:     std::cerr << "GL_OUT_OF_MEMORY";     break;
				default: fprintf(stderr, "UNKNOWN ERROR (0x%04X)", error);      break;
			}
			std::cerr << std::endl;
			found_error = true;
		}

		if (found_error)
			return false;
		else
			return true;
	}
}

#define glCall(call)                       \
{                                          \
	Gra::glClearError();               \
	call;                              \
	assert(Gra::glLogCall() && #call); \
}

#endif
