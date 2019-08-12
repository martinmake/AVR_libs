#ifndef _GPL_PRIMITIVE_BASE_H_
#define _GPL_PRIMITIVE_BASE_H_

#include <assert.h>

#include <gra/math.h>
#include <gra/renderer.h>
#include <gra/shader.h>
#include <gra/vertex_array.h>
#include <gra/vertex_buffer.h>

#include "gpl/static_initializer.h"

namespace Gpl
{
	namespace Primitive
	{
		class Base : private StaticInitializer
		{
			protected:
				Gra::Math::vec4<float> m_color;
			protected:
				Gra::VertexArray  m_vertex_array;
				Gra::VertexBuffer m_vertex_buffer;

			public:
				Base(const Gra::Math::vec4<float>& initial_color);
				Base(Base&& other);
				virtual ~Base(void);

			public:
				virtual void draw(const Gra::Renderer& renderer, const glm::mat4& mvp) const;
		};

		inline void Base::draw(const Gra::Renderer& renderer, const glm::mat4& mvp) const { (void) renderer; (void) mvp; assert(false && "THIS IS JUST AN INTERFACE!"); }
	}
}

#endif
