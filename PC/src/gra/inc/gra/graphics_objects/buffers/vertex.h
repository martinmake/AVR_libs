#ifndef _GRA_GRAPHICS_OBJECT_BUFFER_VERTEX_H_
#define _GRA_GRAPHICS_OBJECT_BUFFER_VERTEX_H_

#include <inttypes.h>
#include <memory>
#include <vector>

#include "gra/gra.h"
#include "gra/graphics_objects/buffers/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Buffer
		{
			class Vertex : public Buffer::Base
			{
				public:
					Vertex(void);
					Vertex(const void* initial_data, uint32_t initial_size);

				public: // GETTERS
					const void* data(void) const; uint32_t size(void) const;
				public: // SETTERS
					void data(const void* new_data, uint32_t new_size);

				public:
					struct Layout
					{
						struct Element
						{
							unsigned int  type;
							unsigned int  count;
							unsigned char normalized;
						};

						std::vector<Element> elements;
						unsigned int         stride = 0;

						template <typename T>
						void push(uint32_t count);
					};

				private:
					void*    m_data;
					uint32_t m_size;

				DECLARATION_MANDATORY(Vertex)
			};

			// GETTERS
			inline const void*    Vertex::data(void) const { return m_data; }
			inline       uint32_t Vertex::size(void) const { return m_size; }

			template <typename T> void Vertex::Layout::push(uint32_t count)
			{
				(void) count;
				static_assert(sizeof(T) && false, "[VERTEX LAYOUT] TYPE UNAVAIBLE!");
			}
			template <> void Vertex::Layout::push<         float>(uint32_t count);
			template <> void Vertex::Layout::push<unsigned int  >(uint32_t count);
			template <> void Vertex::Layout::push<unsigned char >(uint32_t count);

			DEFINITION_MANDATORY(Vertex, )
		}
	}
}

#endif
