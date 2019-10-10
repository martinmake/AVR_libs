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
				: Base(GL_ARRAY_BUFFER)
			{
				TRACE("BUFFER: VERTEX: CONSTRUCTED: {0}", (void*) this);
			}
			Vertex::Vertex(size_t initial_size)
				: Base(GL_ARRAY_BUFFER, initial_size)
			{
				TRACE("BUFFER: VERTEX: CONSTRUCTED: {0}", (void*) this);
			}
			Vertex::Vertex(const void* initial_data, size_t initial_size)
				: Base(GL_ARRAY_BUFFER, initial_data, initial_size)
			{
				TRACE("BUFFER: VERTEX: CONSTRUCTED: {0}", (void*) this);
			}
			Vertex::Vertex(const std::vector<const void*>& initial_data, const std::vector<size_t>& initial_size)
				: Vertex()
			{
				assert(initial_data.size() == initial_size.size());

				for (size_t size : initial_size)
					m_size += size;
				size(m_size);

				std::vector<const void*>::const_iterator initial_data_it = initial_data.begin();
				std::vector<size_t     >::const_iterator initial_size_it = initial_size.begin();
				while (initial_data_it != initial_data.end())
				{
					static size_t data_offset = 0;
					data(*initial_data_it, *initial_size_it, data_offset);

					data_offset += *initial_size_it;

					initial_data_it++;
					initial_size_it++;
				}
			}

			Vertex::~Vertex(void)
			{
				TRACE("BUFFER: VERTEX: DESTRUCTED: {0}", (void*) this);
			}

			void Vertex::copy(const Vertex& other)
			{
				Buffer::Base::copy(other);
			}
			void Vertex::move(Vertex&& other)
			{
				Buffer::Base::move(std::move(other));
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
