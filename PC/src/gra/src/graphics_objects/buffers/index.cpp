#include "logging.h"

#include "gra/graphics_objects/buffers/index.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Buffer
		{
			Index::Index(void)
				: Base(GL_ELEMENT_ARRAY_BUFFER)
			{
				TRACE("BUFFER: INDEX: CONSTRUCTED: {0}", (void*) this);
			}
			Index::Index(const std::vector<type>& initial_indices)
				: Index()
			{
				indices(initial_indices);
			}

			Index::~Index(void)
			{
				TRACE("BUFFER: INDEX: DESTRUCTED: {0}", (void*) this);
			}

			// SETTERS
			void Index::indices(const std::vector<type>& new_indices)
			{
				m_indices = new_indices;
				buffer_data(m_indices.data(), m_indices.size() * sizeof(type));
			}

			void Index::copy(const Index& other)
			{
				Buffer::Base::copy(other);

				indices(other.m_indices);
			}
			void Index::move(Index&& other)
			{
				Buffer::Base::move(std::move(other));

				m_indices = std::move(other.m_indices);
			}
		}
	}
}
