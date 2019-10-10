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

	extern bool glLogCall(void);
}

#define glCall(call)                       \
{                                          \
	Gra::glClearError();               \
	call;                              \
	assert(Gra::glLogCall() && #call); \
}

#endif
