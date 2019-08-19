#include <string.h>
#include <utility>

#include "logging.h"

#include "gra/graphics_objects/buffers/vertex.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Buffer
		{
			Vertex::Vertex(void)
				: Base(GL_ARRAY_BUFFER), m_data(nullptr), m_size(0)
			{
				TRACE("BUFFER: VERTEX: CONSTRUCTED: {0}", (void*) this);
			}
			Vertex::Vertex(const void* initial_data, uint32_t initial_size)
				: Vertex()
			{
				data(initial_data, initial_size);
			}

			Vertex::~Vertex(void)
			{
				free(m_data);
				TRACE("BUFFER: VERTEX: DESTRUCTED: {0}", (void*) this);
			}

			void Vertex::copy(const Vertex& other)
			{
				Buffer::Base::copy(other);

				data(other.m_data, other.m_size);
			}
			void Vertex::move(Vertex&& other)
			{
				Buffer::Base::move(std::move(other));

				free(m_data);
				m_data = std::exchange(other.m_data, nullptr);
				m_size = std::exchange(other.m_size, 0      );
			}

			// SETTERS
			void Vertex::data(const void* new_data, uint32_t new_size)
			{
				if (m_size != new_size)
				{
					free(m_data);
					m_size = new_size;
					m_data = malloc(m_size);
				}
				memcpy(m_data, new_data, m_size);

				buffer_data(m_data, m_size);
			}

			template <>
			void Vertex::Layout::push<float>(uint32_t count)
			{
				elements.push_back({ GL_FLOAT, count, GL_FALSE });
				stride += count * glSizeOf(GL_FLOAT);
			}
			template <>
			void Vertex::Layout::push<unsigned int>(uint32_t count)
			{
				elements.push_back({ GL_UNSIGNED_INT, count, GL_FALSE });
				stride += count * glSizeOf(GL_UNSIGNED_INT);
			}
			template <>
			void Vertex::Layout::push<unsigned char>(uint32_t count)
			{
				elements.push_back({ GL_UNSIGNED_BYTE, count, GL_FALSE });
				stride += count * glSizeOf(GL_UNSIGNED_BYTE);
			}
		}
	}
}
