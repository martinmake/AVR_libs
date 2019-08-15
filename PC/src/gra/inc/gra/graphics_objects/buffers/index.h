#ifndef _GRA_GRAPHICS_OBJECT_BUFFER_INDEX_H_
#define _GRA_GRAPHICS_OBJECT_BUFFER_INDEX_H_

#include <inttypes.h>
#include <utility>
#include <vector>

#include "gra/glstd.h"
#include "gra/gldebug.h"

#include "gra/graphics_objects/buffers/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Buffer
		{
			template <typename T>
			class Index : public Buffer::Base
			{
				private:
					std::vector<T> m_indices;

				public:
					Index(void);
					Index(const std::vector<T>& initial_indices);

					Index(const Index&  other);
					Index(      Index&& other);

					~Index(void);

				public:
					void copy(const Index&  other);
					void move(      Index&& other);

				public: // OPERATORS
					Index& operator=(const Index&  rhs);
					Index& operator=(      Index&& rhs);

				public: // GETTERS
					const std::vector<T>& indices(void) const;
				public: // SETTERS
					void indices(const std::vector<T>& new_indices);
			};

			template <typename T>
			Index<T>::Index(void)
				: Base(GL_ELEMENT_ARRAY_BUFFER)
			{
			}
			template <typename T>
			Index<T>::Index(const std::vector<T>& initial_indices)
				: Index()
			{
				indices(initial_indices);
			}

			template <typename T> Index<T>::Index(const Index&  other) : Base() { copy(          other ); }
			template <typename T> Index<T>::Index(      Index&& other) : Base() { move(std::move(other)); }

			template <typename T>
			Index<T>::~Index(void)
			{
			}

			template <typename T>
			void Index<T>::copy(const Index<T>& other)
			{
				Buffer::Base::copy(other);

				indices(other.m_indices);
			}
			template <typename T>
			void Index<T>::move(Index<T>&& other)
			{
				Buffer::Base::move(std::move(other));

				m_indices = std::move(other.m_indices);
			}

			// OPERATORS
			template <typename T> Index<T>& Index<T>::operator=(const Index&  rhs) { copy(          rhs ); return *this; }
			template <typename T> Index<T>& Index<T>::operator=(      Index&& rhs) { move(std::move(rhs)); return *this; }

			// GETTERS
			template <typename T> const std::vector<T>& Index<T>::indices(void) const { return m_indices; }
			// SETTERS
			template <typename T>
			void Index<T>::indices(const std::vector<T>& new_indices)
			{
				m_indices = new_indices;
				buffer_data(m_indices.data(), m_indices.size() * sizeof(T));
			}

			template class Index<uint8_t >;
			template class Index<uint16_t>;
			template class Index<uint32_t>;
			template class Index<uint64_t>;
		}
	}
}

#endif
