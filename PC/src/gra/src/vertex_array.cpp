#include "glstd.h"
#include "vertex_array.h"
#include "vertex_buffer_layout.h"

namespace Gra
{
	VertexArray::VertexArray(void)
	{
		glCall(glGenVertexArrays(1, &m_renderer_id));
	}
	VertexArray::VertexArray(const VertexBuffer& initial_vertex_buffer, const VertexBufferLayout& initial_layout)
		: VertexArray()
	{
		vertex_buffer(initial_vertex_buffer);
		layout(initial_layout);
	}

	VertexArray::~VertexArray(void)
	{
		glCall(glDeleteVertexArrays(1, &m_renderer_id));
	}

	// SETTERS
	void VertexArray::vertex_buffer(const VertexBuffer& new_vertex_buffer)
	{
		bind();
		new_vertex_buffer.bind();
	}
	void VertexArray::layout(const VertexBufferLayout& new_layout)
	{
		bind();
		const std::vector<VertexBufferLayoutElement>& elements = new_layout.elements();
		uint64_t offset = 0;

		for (uint32_t i = 0; i < elements.size(); i++)
		{
			const VertexBufferLayoutElement& element = elements[i];

			glCall(glEnableVertexAttribArray(i));
			glCall(glVertexAttribPointer(i, element.count, element.type, element.normalized, new_layout.stride(), (const void*) offset));

			offset += glSizeOf(element.type) * element.count;
		}
	}
}
