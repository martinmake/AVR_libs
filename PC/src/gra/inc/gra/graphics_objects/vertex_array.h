#ifndef _GRA_GRAPHICS_OBJECT_VERTEX_ARRAY_H_
#define _GRA_GRAPHICS_OBJECT_VERTEX_ARRAY_H_

#include <inttypes.h>
#include <vector>

#include "gra/gra.h"
#include "gra/window.h"
#include "gra/graphics_objects/buffers/vertex.h"

namespace Gra
{
	namespace GraphicsObject
	{
		class VertexArray : public GraphicsObject::Base
		{
			public:
				VertexArray(const Window& window);
				VertexArray(const Buffer::Vertex& initial_vertex_buffer, const Buffer::Vertex::Layout& initial_vertex_buffer_layout, const Window& window);

			public:
				void   bind(void) const override;
				void unbind(void) const override;

			public: // SETTERS
				void vertex_buffer(const Buffer::Vertex        & new_vertex_buffer);
				void layout       (const Buffer::Vertex::Layout& new_layout       );

			private:
				Buffer::Vertex         m_vertex_buffer;
				Buffer::Vertex::Layout m_vertex_buffer_layout;
				const Window&          m_window;

			DECLARATION_MANDATORY(VertexArray)

		};

		inline void VertexArray::  bind(void) const { m_window.make_current(); glCall(glBindVertexArray(m_renderer_id)); }
		inline void VertexArray::unbind(void) const { m_window.make_current(); glCall(glBindVertexArray(            0)); }

		DEFINITION_MANDATORY(VertexArray, other.m_window)
	}
}

#endif
