#ifndef _GRA_GRAPHICS_OBJECT_BUFFER_VERTEX_H_
#define _GRA_GRAPHICS_OBJECT_BUFFER_VERTEX_H_

#include <inttypes.h>
#include <memory>
#include <vector>

#include "gra/core.h"
#include "gra/graphics_objects/buffers/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Buffer
		{
			class Vertex : public Buffer::Base
			{
				public: // CONSTRUCTORS
					Vertex(void);
					Vertex(size_t initial_size);
					Vertex(const void* initial_data, size_t initial_size);
					Vertex(const std::vector<const void*>& initial_data, const std::vector<size_t>& initial_size);

				public: // TYPES
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

				DECLARATION_MANDATORY(Vertex)
			};

			// FUNCTIONS
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
