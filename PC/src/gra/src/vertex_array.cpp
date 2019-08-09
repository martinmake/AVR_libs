#include "glstd.h"
#include "vertex_array.h"
#include "vertex_buffer_layout.h"

namespace Gra
{
	VertexArray::VertexArray(void)
	{
		glCall(glGenVertexArrays(1, &m_renderer_id));
	}

	VertexArray::~VertexArray(void)
	{
		glCall(glDeleteVertexArrays(1, &m_renderer_id));
	}

	void VertexArray::add_buffer(const VertexBuffer& vertex_buffer, const VertexBufferLayout& layout)
	{
		bind();
		vertex_buffer.bind();

		const std::vector<VertexBufferLayoutElement>& elements = layout.elements();
		uint64_t offset = 0;

		for (uint32_t i = 0; i < elements.size(); i++)
		{
			const VertexBufferLayoutElement& element = elements[i];

			glCall(glEnableVertexAttribArray(i));
			glCall(glVertexAttribPointer(i, element.count, element.type, element.normalized, layout.stride(), (const void*) offset));

			offset += glSizeOf(element.type) * element.count;
		}
	}
}
