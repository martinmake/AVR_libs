#include "gra/vertex_buffer.h"

namespace Gra
{
	VertexBuffer::VertexBuffer(void)
	{
		glCall(glGenBuffers(1, &m_renderer_id));
	}
	VertexBuffer::VertexBuffer(const void* initial_data, uint32_t size)
		: VertexBuffer()
	{
		data(initial_data, size);
	}

	VertexBuffer::~VertexBuffer(void)
	{
		glCall(glDeleteBuffers(1, &m_renderer_id));
	}

	void VertexBuffer::data(const void* new_data, uint32_t size)
	{
		bind();
		glCall(glBufferData(GL_ARRAY_BUFFER, size, new_data, GL_STATIC_DRAW));
	}
}
