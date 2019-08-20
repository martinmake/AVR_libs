#include "logging.h"

#include "gra/graphics_objects/vertex_array.h"

namespace Gra
{
	namespace GraphicsObject
	{
		VertexArray::VertexArray(void)
		{
			glCall(glGenVertexArrays(1, &m_renderer_id));
			TRACE("GL: VERTEX ARRAY: GENERATED: {0}", m_renderer_id);
			TRACE("VERTEX ARRAY: CONSTRUCTED: {0}", (void*) this);
		}
		VertexArray::VertexArray(const Buffer::Vertex& initial_vertex_buffer, const Buffer::Vertex::Layout& initial_vertex_buffer_layout)
			: VertexArray()
		{
			vertex_buffer(initial_vertex_buffer);
			layout(initial_vertex_buffer_layout);
		}

		VertexArray::~VertexArray(void)
		{
			if (m_renderer_id)
			{
				glCall(glDeleteVertexArrays(1, &m_renderer_id));
				TRACE("GL: VERTEX ARRAY: DELETED: {0}", m_renderer_id);
			}
			TRACE("VERTEX ARRAY: DESTRUCTED: {0}", (void*) this);
		}

		// SETTERS
		void VertexArray::vertex_buffer(const Buffer::Vertex& new_vertex_buffer)
		{
			m_vertex_buffer = new_vertex_buffer;
			bind();
			m_vertex_buffer.bind();
		}
		void VertexArray::layout(const Buffer::Vertex::Layout& new_vertex_buffer_layout)
		{
			m_vertex_buffer_layout = new_vertex_buffer_layout;
			uint64_t offset = 0;

			bind();
			for (uint32_t i = 0; i < m_vertex_buffer_layout.elements.size(); i++)
			{
				const Buffer::Vertex::Layout::Element& element = m_vertex_buffer_layout.elements[i];

				glCall(glEnableVertexAttribArray(i));
				glCall(glVertexAttribPointer(i, element.count, element.type, element.normalized, m_vertex_buffer_layout.stride, (const void*) offset));

				offset += glSizeOf(element.type) * element.count;
			}
		}
	}
}
