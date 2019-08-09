#include "vertex_buffer_layout.h"

namespace Gra
{
	VertexBufferLayout::VertexBufferLayout(void)
		: m_stride(0)
	{
	}

	VertexBufferLayout::~VertexBufferLayout(void)
	{
	}

	template <>
	void VertexBufferLayout::push<float>(uint32_t count)
	{
		m_elements.push_back({ GL_FLOAT, count, GL_FALSE });
		m_stride += count * glSizeOf(GL_FLOAT);
	}

	template <>
	void VertexBufferLayout::push<unsigned int>(uint32_t count)
	{
		m_elements.push_back({ GL_UNSIGNED_INT, count, GL_FALSE });
		m_stride += count * glSizeOf(GL_UNSIGNED_INT);
	}

	template <>
	void VertexBufferLayout::push<unsigned char>(uint32_t count)
	{
		m_elements.push_back({ GL_UNSIGNED_BYTE, count, GL_FALSE });
		m_stride += count * glSizeOf(GL_UNSIGNED_BYTE);
	}
}
