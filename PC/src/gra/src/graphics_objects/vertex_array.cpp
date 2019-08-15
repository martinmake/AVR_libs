#include "gra/graphics_objects/vertex_array.h"

namespace Gra
{
	namespace GraphicsObject
	{
		VertexArray::VertexArray(void)
		{
			glCall(glGenVertexArrays(1, &m_renderer_id));
		}
		VertexArray::VertexArray(const Buffer::Vertex& initial_vertex_buffer, const Layout& initial_layout)
			: VertexArray()
		{
			vertex_buffer(initial_vertex_buffer);
			layout(initial_layout);
		}

		VertexArray::~VertexArray(void)
		{
			if (m_renderer_id)
				glCall(glDeleteVertexArrays(1, &m_renderer_id));
		}

		// SETTERS
		void VertexArray::vertex_buffer(const Buffer::Vertex& new_vertex_buffer)
		{
			bind();
			new_vertex_buffer.bind();
		}
		void VertexArray::layout(const Layout& new_layout)
		{
			m_layout = new_layout;
			uint64_t offset = 0;

			bind();
			for (uint32_t i = 0; i < m_layout.elements.size(); i++)
			{
				const Layout::Element& element = m_layout.elements[i];

				glCall(glEnableVertexAttribArray(i));
				glCall(glVertexAttribPointer(i, element.count, element.type, element.normalized, m_layout.stride, (const void*) offset));

				offset += glSizeOf(element.type) * element.count;
			}
		}

		template <>
		void VertexArray::Layout::push<float>(uint32_t count)
		{
			elements.push_back({ GL_FLOAT, count, GL_FALSE });
			stride += count * glSizeOf(GL_FLOAT);
		}
		template <>
		void VertexArray::Layout::push<unsigned int>(uint32_t count)
		{
			elements.push_back({ GL_UNSIGNED_INT, count, GL_FALSE });
			stride += count * glSizeOf(GL_UNSIGNED_INT);
		}
		template <>
		void VertexArray::Layout::push<unsigned char>(uint32_t count)
		{
			elements.push_back({ GL_UNSIGNED_BYTE, count, GL_FALSE });
			stride += count * glSizeOf(GL_UNSIGNED_BYTE);
		}
	}
}
