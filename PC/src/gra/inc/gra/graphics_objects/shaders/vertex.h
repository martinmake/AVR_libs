#ifndef _GRA_GRAPHICS_OBJECT_SHADER_VERTEX_H_
#define _GRA_GRAPHICS_OBJECT_SHADER_VERTEX_H_

#include "gra/graphics_objects/shaders/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Shader
		{
			class Vertex : public Shader::Base
			{
				public:
					Vertex(void);
					Vertex(const std::string& filepath_or_source);

					Vertex(const Vertex&  other);
					Vertex(      Vertex&& other);

					~Vertex(void);

				private:
					void copy(const Vertex&  other);
					void move(      Vertex&& other);
				public:
					Vertex& operator=(const Vertex&  rhs);
					Vertex& operator=(      Vertex&& rhs);
			};

			inline Vertex::Vertex(const Vertex&  other) : Vertex() { copy(          other ); }
			inline Vertex::Vertex(      Vertex&& other) : Vertex() { move(std::move(other)); }

			inline Vertex& Vertex::operator=(const Vertex&  rhs) { copy(          rhs ); return *this; }
			inline Vertex& Vertex::operator=(      Vertex&& rhs) { move(std::move(rhs)); return *this; }
		}
	}
}

#endif
