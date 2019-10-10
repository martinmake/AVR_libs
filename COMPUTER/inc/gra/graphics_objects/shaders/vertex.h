#ifndef _GRA_GRAPHICS_OBJECT_SHADER_VERTEX_H_
#define _GRA_GRAPHICS_OBJECT_SHADER_VERTEX_H_

#include "gra/gra.h"
#include "gra/graphics_objects/shaders/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			class Vertex : public Shader::Base
			{
				public: // CONSTRUCTORS
					Vertex(void);
					Vertex(const std::string& filepath_or_source);

				DECLARATION_MANDATORY(Vertex)
			};

			DEFINITION_MANDATORY(Vertex, )
		}
	}
}

#endif
