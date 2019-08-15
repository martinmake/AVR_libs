#ifndef _GRA_GRAPHICS_OBJECT_BUFFER_VERTEX_H_
#define _GRA_GRAPHICS_OBJECT_BUFFER_VERTEX_H_

#include <inttypes.h>
#include <memory>

#include "gra/glstd.h"
#include "gra/gldebug.h"

#include "gra/graphics_objects/buffers/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Buffer
		{
			class Vertex : public Buffer::Base
			{
				private:
					void*    m_data;
					uint32_t m_size;

				public:
					Vertex(void);
					Vertex(const void* initial_data, uint32_t initial_size);

					Vertex(const Vertex&  other);
					Vertex(      Vertex&& other);

					~Vertex(void);

				public:
					void copy(const Vertex&  other);
					void move(      Vertex&& other);

				public: // OPERATORS
					Vertex& operator=(const Vertex&  rhs);
					Vertex& operator=(      Vertex&& rhs);

				public: // GETTERS
					const void*    data(void) const;
					      uint32_t size(void) const;
				public: // SETTERS
					void data(const void* new_data, uint32_t new_size);
			};

			inline Vertex::Vertex(const Vertex&  other) : Base() { copy(          other ); }
			inline Vertex::Vertex(      Vertex&& other) : Base() { move(std::move(other)); }

			// OPERATORS
			inline Vertex& Vertex::operator=(const Vertex&  rhs) { copy(          rhs ); return *this; }
			inline Vertex& Vertex::operator=(      Vertex&& rhs) { move(std::move(rhs)); return *this; }

			// GETTERS
			inline const void*    Vertex::data(void) const { return m_data; }
			inline       uint32_t Vertex::size(void) const { return m_size; }
		}
	}
}

#endif
