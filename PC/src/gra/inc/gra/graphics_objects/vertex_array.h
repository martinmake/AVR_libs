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

			public: // GETTERS
				      uint32_t                count               (void) const;
				      Buffer::Vertex        & vertex_buffer       (void);
				const Buffer::Vertex::Layout& vertex_buffer_layout(void) const;
			public: // SETTERS
				void vertex_buffer(const Buffer::Vertex        & new_vertex_buffer       );
				void layout       (const Buffer::Vertex::Layout& new_vertex_buffer_layout);

			private:
				Buffer::Vertex         m_vertex_buffer;
				Buffer::Vertex::Layout m_vertex_buffer_layout;

			DECLARATION_MANDATORY(VertexArray)
		};

		// GETTERS
		inline uint32_t VertexArray::count(void) const { return m_vertex_buffer.size() / m_vertex_buffer_layout.stride; }
		//
		inline Buffer::Vertex& VertexArray::vertex_buffer(void) { return m_vertex_buffer; }
		//
		DEFINITION_DEFAULT_GETTER(VertexArray, vertex_buffer_layout, const Buffer::Vertex::Layout&)

		// FUNCTIONS
		inline void VertexArray::  bind(void) const { glCall(glBindVertexArray(m_renderer_id)); }
		inline void VertexArray::unbind(void) const { glCall(glBindVertexArray(            0)); }

		DEFINITION_MANDATORY(VertexArray, )
	}
}

#endif
