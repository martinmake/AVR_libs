#include <string.h>
#include <utility>

#include "gra/graphics_objects/buffers/vertex.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Buffer
		{
			Vertex::Vertex(void)
				: Base(GL_ARRAY_BUFFER)
			{
			}
			Vertex::Vertex(const void* initial_data, uint32_t initial_size)
				: Vertex()
			{
				data(initial_data, initial_size);
			}

			Vertex::~Vertex(void)
			{
				free(m_data);
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
		}
	}
}
