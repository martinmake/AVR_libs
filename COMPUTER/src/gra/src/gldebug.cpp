#include "logging.h"

#include "gra/gldebug.h"

namespace Gra
{
	bool glLogCall(void)
	{
		bool found_error = false;

		while (GLenum error = glGetError())
		{
			std::string description;
			switch (error)
			{
				case GL_INVALID_ENUM:      description = "GL_INVALID_ENUM";      break;
				case GL_INVALID_VALUE:     description = "GL_INVALID_VALUE";     break;
				case GL_INVALID_OPERATION: description = "GL_INVALID_OPERATION"; break;
				case GL_STACK_OVERFLOW:    description = "GL_STACK_OVERFLOW";    break;
				case GL_STACK_UNDERFLOW:   description = "GL_STACK_UNDERFLOW";   break;
				case GL_OUT_OF_MEMORY:     description = "GL_OUT_OF_MEMORY";     break;
				default:                   description = "UNKNOWN";              break;
			}
			ERROR("GL: ERROR: {0} ({1:x})", description.c_str(), error)

			found_error = true;
		}

		if (found_error)
			return false;
		else
			return true;
	}
}

