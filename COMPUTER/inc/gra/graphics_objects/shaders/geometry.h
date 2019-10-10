#ifndef _GRA_GRAPHICS_OBJECT_SHADER_GEOMETRY_H_
#define _GRA_GRAPHICS_OBJECT_SHADER_GEOMETRY_H_

#include "gra/gra.h"
#include "gra/graphics_objects/shaders/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			class Geometry : public Shader::Base
			{
				public: // CONSTRUCTORS
					Geometry(void);
					Geometry(const std::string& filepath_or_source);

				DECLARATION_MANDATORY(Geometry)
			};

			DEFINITION_MANDATORY(Geometry, )
		}
	}
}

#endif
