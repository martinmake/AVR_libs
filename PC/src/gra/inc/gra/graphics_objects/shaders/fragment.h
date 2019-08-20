#ifndef _GRA_GRAPHICS_OBJECT_SHADER_FRAGMENT_H_
#define _GRA_GRAPHICS_OBJECT_SHADER_FRAGMENT_H_

#include "gra/gra.h"
#include "gra/graphics_objects/shaders/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			class Fragment : public Shader::Base
			{
				public: // CONSTRUCTORS
					Fragment(void);
					Fragment(const std::string& filepath_or_source);

				DECLARATION_MANDATORY(Fragment)
			};

			DEFINITION_MANDATORY(Fragment, )
		}
	}
}

#endif
