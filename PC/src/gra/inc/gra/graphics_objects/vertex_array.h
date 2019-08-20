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
			public: // CONSTRUCTORS
				VertexArray(void);
				VertexArray(const Buffer::Vertex& initial_vertex_buffer, const Buffer::Vertex::Layout& initial_vertex_buffer_layout);

			public: // FUNCTIONS
				void   bind(void) const override;
				void unbind(void) const override;

			public: // SETTERS
				void vertex_buffer(const Buffer::Vertex        & new_vertex_buffer);
				void layout       (const Buffer::Vertex::Layout& new_layout       );

			private:
				Buffer::Vertex         m_vertex_buffer;
				Buffer::Vertex::Layout m_vertex_buffer_layout;

			DECLARATION_MANDATORY(VertexArray)

		};

		// FUNCTIONS
		inline void VertexArray::  bind(void) const { glCall(glBindVertexArray(m_renderer_id)); }
		inline void VertexArray::unbind(void) const { glCall(glBindVertexArray(            0)); }

		DEFINITION_MANDATORY(VertexArray, )
	}
}

#endif
