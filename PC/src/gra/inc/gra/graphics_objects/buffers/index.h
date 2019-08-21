#ifndef _GRA_GRAPHICS_OBJECT_BUFFER_INDEX_H_
#define _GRA_GRAPHICS_OBJECT_BUFFER_INDEX_H_

#include <inttypes.h>
#include <utility>
#include <vector>

#include "gra/gra.h"
#include "gra/graphics_objects/buffers/base.h"

namespace Gra
{
	namespace GraphicsObject
	{
		namespace Buffer
		{
			class Index : public Buffer::Base
			{
				public: // TYPES
					using type = uint32_t;

				public: // CONSTRUCTORS
					Index(void);
					Index(const std::vector<type>& initial_indices);

				public: // GETTERS
					size_t count(void) const;
				public: // SETTERS
					void indices(const std::vector<type>& new_indices);

				private:
					size_t m_count;

				DECLARATION_MANDATORY(Index)
			};

			// GETTERS
			inline size_t Index::count(void) const { return m_count; }
			// SETTERS
			inline void Index::indices(const std::vector<type>& new_indices) { m_count = new_indices.size(); data(new_indices.data(), new_indices.size() * sizeof(type)); }

			DEFINITION_MANDATORY(Index, )
		}
	}
}

#endif
